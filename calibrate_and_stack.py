from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score

from direction_pipeline import (
    DirectionConfig,
    LABELS,
    TECH_FEATURES,
    TemporalConvNetClassifier,
    LogisticRandomForestEnsemble,
    _aggregate_daily,
    _ensure_price_col,
    _sort_df,
    _build_supervised_for_company,
    _progress,
)

TARGET_CLASSES = np.array([-1, 0, 1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate TCN/Ensemble probabilities and run stacked meta-classifier.")
    parser.add_argument("--data-file", default="asset_events_df.csv")
    parser.add_argument("--output-dir", default="analysis_outputs/calibrated_comparison")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calib-size", type=float, default=0.1)
    parser.add_argument("--prediction-margin", type=float, default=0.15)
    parser.add_argument("--margin-min", type=float, default=0.0)
    parser.add_argument("--margin-max", type=float, default=0.2)
    parser.add_argument("--margin-step", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight for macro F1 in joint score.")
    parser.add_argument("--beta", type=float, default=0.4, help="Weight for accuracy in joint score.")
    parser.add_argument("--gamma", type=float, default=0.2, help="Weight for coverage in joint score.")
    parser.add_argument("--n-lags", type=int, default=30)
    parser.add_argument("--tcn-hidden-channels", type=int, default=128)
    parser.add_argument("--tcn-num-layers", type=int, default=4)
    parser.add_argument("--tcn-kernel-size", type=int, default=5)
    parser.add_argument("--tcn-dropout", type=float, default=0.2)
    parser.add_argument("--tcn-lr", type=float, default=5e-4)
    parser.add_argument("--tcn-batch-size", type=int, default=128)
    parser.add_argument("--tcn-max-epochs", type=int, default=80)
    parser.add_argument("--tcn-patience", type=int, default=10)
    parser.add_argument("--tcn-device", default="cuda", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def _time_split(n: int, test_size: float, calib_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_test = max(1, int(round(n * test_size)))
    n_calib = max(1, int(round(n * calib_size)))
    n_train = max(1, n - n_test - n_calib)
    if n_train + n_calib + n_test > n:
        n_test = max(1, n - n_train - n_calib)
    idx = np.arange(n)
    train = idx[:n_train]
    calib = idx[n_train:n_train + n_calib]
    test = idx[n_train + n_calib:]
    return train, calib, test


def _align_proba(proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(TARGET_CLASSES)))
    for i, cls in enumerate(model_classes):
        matches = np.where(TARGET_CLASSES == int(cls))[0]
        if len(matches) == 0:
            continue
        aligned[:, matches[0]] = proba[:, i]
    return aligned


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _temperature_scale(logits: np.ndarray, y: np.ndarray) -> float:
    best_t = 1.0
    best_nll = float('inf')
    for t in np.linspace(0.5, 3.0, 26):
        probs = _softmax(logits / t)
        p = probs[np.arange(len(y)), y]
        nll = -np.mean(np.log(np.clip(p, 1e-12, 1.0)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def _fit_platt(scores: np.ndarray, y: np.ndarray) -> List[LogisticRegression]:
    models: List[LogisticRegression] = []
    for k in range(scores.shape[1]):
        binary = (y == k).astype(int)
        clf = LogisticRegression(solver="lbfgs")
        clf.fit(scores[:, [k]], binary)
        models.append(clf)
    return models


def _apply_platt(models: List[LogisticRegression], scores: np.ndarray) -> np.ndarray:
    calibrated = np.zeros_like(scores)
    for k, clf in enumerate(models):
        calibrated[:, k] = clf.predict_proba(scores[:, [k]])[:, 1]
    return _renormalize(calibrated)


def _fit_isotonic(scores: np.ndarray, y: np.ndarray) -> List[IsotonicRegression]:
    models: List[IsotonicRegression] = []
    for k in range(scores.shape[1]):
        binary = (y == k).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores[:, k], binary)
        models.append(iso)
    return models


def _apply_isotonic(models: List[IsotonicRegression], scores: np.ndarray) -> np.ndarray:
    calibrated = np.zeros_like(scores)
    for k, iso in enumerate(models):
        calibrated[:, k] = iso.predict(scores[:, k])
    return _renormalize(calibrated)


def _renormalize(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-8, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _apply_margin_rule(proba: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    idx_down, idx_flat, idx_up = 0, 1, 2
    p_down = proba[:, idx_down]
    p_flat = proba[:, idx_flat]
    p_up = proba[:, idx_up]
    dir_is_up = p_up >= p_down
    dir_labels = np.where(dir_is_up, 1, -1)
    dir_prob = np.where(dir_is_up, p_up, p_down)
    margin_vals = dir_prob - p_flat
    operable = margin_vals >= margin
    final_pred = np.where(operable, dir_labels, 0)
    return final_pred, operable


def _directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    operable: np.ndarray,
) -> Dict[str, float]:
    non_flat = y_true != 0
    if not np.any(non_flat):
        return {
            "dir_acc": np.nan,
            "dir_f1": np.nan,
            "dir_acc_operable": np.nan,
        }
    dir_acc = float(np.mean(y_pred[non_flat] == y_true[non_flat]))
    dir_f1 = float(f1_score(y_true[non_flat], y_pred[non_flat], labels=[-1, 1], average="macro"))
    operable_mask = non_flat & operable
    dir_acc_operable = float(np.mean(y_pred[operable_mask] == y_true[operable_mask])) if operable_mask.any() else np.nan
    return {
        "dir_acc": dir_acc,
        "dir_f1": dir_f1,
        "dir_acc_operable": dir_acc_operable,
    }


def _regime_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    operable: np.ndarray,
    vol_series: np.ndarray | None,
) -> Dict[str, float]:
    if vol_series is None or len(vol_series) == 0:
        return {
            "high_vol_dir_acc_operable": np.nan,
            "low_vol_dir_acc_operable": np.nan,
            "high_vol_dir_f1": np.nan,
            "low_vol_dir_f1": np.nan,
            "high_vol_coverage": np.nan,
            "low_vol_coverage": np.nan,
        }
    median = float(np.nanmedian(vol_series))
    high_mask = vol_series >= median
    low_mask = vol_series < median

    def _subset(mask: np.ndarray) -> Dict[str, float]:
        if not np.any(mask):
            return {
                "dir_acc_operable": np.nan,
                "dir_f1": np.nan,
                "coverage": np.nan,
            }
        metrics = _directional_metrics(y_true[mask], y_pred[mask], operable[mask])
        coverage = float(np.mean(operable[mask]))
        return {
            "dir_acc_operable": metrics["dir_acc_operable"],
            "dir_f1": metrics["dir_f1"],
            "coverage": coverage,
        }

    high = _subset(high_mask)
    low = _subset(low_mask)
    return {
        "high_vol_dir_acc_operable": high["dir_acc_operable"],
        "low_vol_dir_acc_operable": low["dir_acc_operable"],
        "high_vol_dir_f1": high["dir_f1"],
        "low_vol_dir_f1": low["dir_f1"],
        "high_vol_coverage": high["coverage"],
        "low_vol_coverage": low["coverage"],
    }


def _evaluate_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    margin: float,
    vol_series: np.ndarray | None = None,
) -> Dict[str, float]:
    preds, operable = _apply_margin_rule(proba, margin)
    acc = accuracy_score(y_true, preds)
    f1m = f1_score(y_true, preds, average="macro")
    operable_share = float(np.mean(operable)) if len(operable) else np.nan
    operable_acc = float(np.mean((preds == y_true)[operable])) if operable.any() else np.nan
    operable_f1 = (
        float(f1_score(y_true[operable], preds[operable], average="macro"))
        if operable.any()
        else np.nan
    )
    metrics = {
        "acc_test": float(acc),
        "f1_macro_test": float(f1m),
        "operable_share": operable_share,
        "operable_accuracy": operable_acc,
        "operable_f1": operable_f1,
    }
    metrics.update(_directional_metrics(y_true, preds, operable))
    metrics.update(_regime_metrics(y_true, preds, operable, vol_series))
    return metrics


def _score_metrics(metrics: Dict[str, float], alpha: float, beta: float, gamma: float) -> float:
    return (
        alpha * metrics["f1_macro_test"]
        + beta * metrics["acc_test"]
        + gamma * metrics["operable_share"]
    )


def _margin_grid(mn: float, mx: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("margin-step must be positive")
    values = list(np.arange(mn, mx + 1e-9, step))
    return [float(v) for v in values]


def _get_tcn_logits(model: TemporalConvNetClassifier, X: np.ndarray) -> np.ndarray:
    logits = model._predict_logits(X)
    return logits


def _get_ensemble_logits(model: LogisticRandomForestEnsemble, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return np.log(np.clip(proba, 1e-12, 1.0))


def main() -> None:
    args = _parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_file, parse_dates=["date"])
    df["company"] = df["company"].astype("category")

    cfg = DirectionConfig(
        horizons=(1, 2, 3, 4, 5),
        n_lags=args.n_lags,
        flat_threshold_pct=0.1,
        flat_threshold_strategy="atr_sqrt",
        atr_threshold_multiplier=0.5,
        atr_up_multiplier=0.6,
        atr_down_multiplier=0.5,
        prediction_margin_threshold=args.prediction_margin,
        balance_classes=True,
        use_technical_indicators=True,
        aggregate_daily=True,
    )

    if cfg.aggregate_daily:
        df = _aggregate_daily(df, cfg)
    df = _ensure_price_col(df, cfg.price_col)
    df = _sort_df(df, cfg.company_col, cfg.date_col)

    companies = df[cfg.company_col].dropna().unique().tolist()
    results: List[Dict[str, float | int | str]] = []
    sweep_rows: List[Dict[str, float | int | str]] = []
    margin_grid = _margin_grid(args.margin_min, args.margin_max, args.margin_step)

    for h in _progress(cfg.horizons, desc="Horizons"):
        for comp in _progress(companies, desc=f"H{h} companies", total=len(companies)):
            cdf = df[df[cfg.company_col] == comp].reset_index(drop=True)
            if len(cdf) < (cfg.n_lags + h + 20):
                continue
            per_h = _build_supervised_for_company(cdf, cfg)[h]
            if per_h.empty:
                continue

            feature_cols = [c for c in per_h.columns if c.startswith("lag_") or c in TECH_FEATURES]
            X = per_h[feature_cols].values
            y = per_h["y"].values
            vol_series = None
            if "atr_14" in per_h.columns and "lag_1" in per_h.columns:
                base = per_h["lag_1"].replace(0, np.nan).astype(float)
                vol_series = (per_h["atr_14"].astype(float) / base).to_numpy()

            train_idx, calib_idx, test_idx = _time_split(len(per_h), args.test_size, args.calib_size)
            if len(test_idx) == 0 or len(calib_idx) == 0:
                continue

            tcn = TemporalConvNetClassifier(
                balance_classes=True,
                hidden_channels=args.tcn_hidden_channels,
                num_layers=args.tcn_num_layers,
                kernel_size=args.tcn_kernel_size,
                dropout=args.tcn_dropout,
                lr=args.tcn_lr,
                batch_size=args.tcn_batch_size,
                max_epochs=args.tcn_max_epochs,
                patience=args.tcn_patience,
                device=args.tcn_device,
            )
            ensemble = LogisticRandomForestEnsemble(balance_classes=True, logistic_weight=0.5)

            tcn.fit(X[train_idx], y[train_idx])
            ensemble.fit(X[train_idx], y[train_idx])

            # Raw probabilities
            tcn_proba_calib = _align_proba(tcn.predict_proba(X[calib_idx]), tcn.classes_)
            tcn_proba_test = _align_proba(tcn.predict_proba(X[test_idx]), tcn.classes_)
            ens_proba_calib = _align_proba(ensemble.predict_proba(X[calib_idx]), ensemble.classes_)
            ens_proba_test = _align_proba(ensemble.predict_proba(X[test_idx]), ensemble.classes_)

            # Raw evaluation
            for model_name, proba_test in [
                ("tcn", tcn_proba_test),
                ("ensemble", ens_proba_test),
            ]:
                metrics = _evaluate_predictions(
                    y[test_idx],
                    proba_test,
                    cfg.prediction_margin_threshold,
                    None if vol_series is None else vol_series[test_idx],
                )
                for margin in margin_grid:
                    m = _evaluate_predictions(
                        y[test_idx],
                        proba_test,
                        margin,
                        None if vol_series is None else vol_series[test_idx],
                    )
                    sweep_rows.append({
                        "horizon": h,
                        "company": comp,
                        "model": model_name,
                        "calibration": "raw",
                        "margin": float(margin),
                        "score": _score_metrics(m, args.alpha, args.beta, args.gamma),
                        **m,
                    })
                results.append({
                    "horizon": h,
                    "company": comp,
                    "model": model_name,
                    "calibration": "raw",
                    **metrics,
                })

            # Temperature scaling (multi-class)
            tcn_logits_calib = _align_proba(_get_tcn_logits(tcn, X[calib_idx]), tcn.classes_)
            tcn_logits_test = _align_proba(_get_tcn_logits(tcn, X[test_idx]), tcn.classes_)
            ens_logits_calib = _align_proba(_get_ensemble_logits(ensemble, X[calib_idx]), ensemble.classes_)
            ens_logits_test = _align_proba(_get_ensemble_logits(ensemble, X[test_idx]), ensemble.classes_)

            tcn_t = _temperature_scale(tcn_logits_calib, np.searchsorted(TARGET_CLASSES, y[calib_idx]))
            ens_t = _temperature_scale(ens_logits_calib, np.searchsorted(TARGET_CLASSES, y[calib_idx]))

            tcn_temp = _softmax(tcn_logits_test / tcn_t)
            ens_temp = _softmax(ens_logits_test / ens_t)

            for model_name, proba_test in [
                ("tcn", tcn_temp),
                ("ensemble", ens_temp),
            ]:
                metrics = _evaluate_predictions(
                    y[test_idx],
                    proba_test,
                    cfg.prediction_margin_threshold,
                    None if vol_series is None else vol_series[test_idx],
                )
                for margin in margin_grid:
                    m = _evaluate_predictions(
                        y[test_idx],
                        proba_test,
                        margin,
                        None if vol_series is None else vol_series[test_idx],
                    )
                    sweep_rows.append({
                        "horizon": h,
                        "company": comp,
                        "model": model_name,
                        "calibration": "temperature",
                        "margin": float(margin),
                        "score": _score_metrics(m, args.alpha, args.beta, args.gamma),
                        **m,
                    })
                results.append({
                    "horizon": h,
                    "company": comp,
                    "model": model_name,
                    "calibration": "temperature",
                    **metrics,
                })

            # Platt scaling and isotonic (one-vs-rest)
            y_calib_idx = np.searchsorted(TARGET_CLASSES, y[calib_idx])
            tcn_platt_models = _fit_platt(tcn_logits_calib, y_calib_idx)
            ens_platt_models = _fit_platt(ens_logits_calib, y_calib_idx)
            tcn_iso_models = _fit_isotonic(tcn_logits_calib, y_calib_idx)
            ens_iso_models = _fit_isotonic(ens_logits_calib, y_calib_idx)

            tcn_platt_test = _apply_platt(tcn_platt_models, tcn_logits_test)
            ens_platt_test = _apply_platt(ens_platt_models, ens_logits_test)

            tcn_iso_test = _apply_isotonic(tcn_iso_models, tcn_logits_test)
            ens_iso_test = _apply_isotonic(ens_iso_models, ens_logits_test)

            for model_name, proba_test, method in [
                ("tcn", tcn_platt_test, "platt"),
                ("ensemble", ens_platt_test, "platt"),
                ("tcn", tcn_iso_test, "isotonic"),
                ("ensemble", ens_iso_test, "isotonic"),
            ]:
                metrics = _evaluate_predictions(
                    y[test_idx],
                    proba_test,
                    cfg.prediction_margin_threshold,
                    None if vol_series is None else vol_series[test_idx],
                )
                for margin in margin_grid:
                    m = _evaluate_predictions(
                        y[test_idx],
                        proba_test,
                        margin,
                        None if vol_series is None else vol_series[test_idx],
                    )
                    sweep_rows.append({
                        "horizon": h,
                        "company": comp,
                        "model": model_name,
                        "calibration": method,
                        "margin": float(margin),
                        "score": _score_metrics(m, args.alpha, args.beta, args.gamma),
                        **m,
                    })
                results.append({
                    "horizon": h,
                    "company": comp,
                    "model": model_name,
                    "calibration": method,
                    **metrics,
                })

            # Stacked meta-classifier on logits
            meta_X_calib = np.concatenate([tcn_logits_calib, ens_logits_calib], axis=1)
            meta_X_test = np.concatenate([tcn_logits_test, ens_logits_test], axis=1)
            meta = LogisticRegression(max_iter=2000, solver="lbfgs")
            meta.fit(meta_X_calib, y_calib_idx)
            meta_proba = meta.predict_proba(meta_X_test)
            metrics = _evaluate_predictions(
                y[test_idx],
                meta_proba,
                cfg.prediction_margin_threshold,
                None if vol_series is None else vol_series[test_idx],
            )
            for margin in margin_grid:
                m = _evaluate_predictions(
                    y[test_idx],
                    meta_proba,
                    margin,
                    None if vol_series is None else vol_series[test_idx],
                )
                sweep_rows.append({
                    "horizon": h,
                    "company": comp,
                    "model": "stacked",
                    "calibration": "meta",
                    "margin": float(margin),
                    "score": _score_metrics(m, args.alpha, args.beta, args.gamma),
                    **m,
                })
            results.append({
                "horizon": h,
                "company": comp,
                "model": "stacked",
                "calibration": "meta",
                **metrics,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(outdir / "per_company_metrics.csv", index=False)

    summary = (
        results_df
        .groupby(["horizon", "model", "calibration"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["model", "calibration", "horizon"])
    )
    summary.to_csv(outdir / "metrics_summary.csv", index=False)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(outdir / "margin_sweep.csv", index=False)

    sweep_summary = (
        sweep_df
        .groupby(["horizon", "model", "calibration", "margin"], as_index=False)
        .mean(numeric_only=True)
    )
    sweep_summary.to_csv(outdir / "margin_sweep_summary.csv", index=False)

    best_joint = (
        sweep_summary
        .sort_values(["horizon", "model", "calibration", "score"], ascending=[True, True, True, False])
        .groupby(["horizon", "model", "calibration"], as_index=False)
        .head(1)
        .assign(selection="best_joint")
    )
    best_f1 = (
        sweep_summary
        .sort_values(["horizon", "model", "calibration", "f1_macro_test"], ascending=[True, True, True, False])
        .groupby(["horizon", "model", "calibration"], as_index=False)
        .head(1)
        .assign(selection="best_f1")
    )
    best_acc = (
        sweep_summary
        .sort_values(["horizon", "model", "calibration", "acc_test"], ascending=[True, True, True, False])
        .groupby(["horizon", "model", "calibration"], as_index=False)
        .head(1)
        .assign(selection="best_acc")
    )
    sweep_best = pd.concat([best_joint, best_f1, best_acc], ignore_index=True)
    sweep_best.to_csv(outdir / "margin_sweep_best.csv", index=False)

    def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        rows = df.to_dict("records")
        for i, a in enumerate(rows):
            dominated = False
            for j, b in enumerate(rows):
                if i == j:
                    continue
                if (
                    b["f1_macro_test"] >= a["f1_macro_test"]
                    and b["acc_test"] >= a["acc_test"]
                    and b["operable_share"] >= a["operable_share"]
                    and (
                        b["f1_macro_test"] > a["f1_macro_test"]
                        or b["acc_test"] > a["acc_test"]
                        or b["operable_share"] > a["operable_share"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                keep.append(a)
        return pd.DataFrame(keep)

    pareto_rows = []
    for (h, model, calibration), group in sweep_summary.groupby(["horizon", "model", "calibration"], as_index=False):
        frontier = _pareto_front(group)
        frontier["horizon"] = h
        frontier["model"] = model
        frontier["calibration"] = calibration
        pareto_rows.append(frontier)
    pareto_df = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else pd.DataFrame()
    pareto_df.to_csv(outdir / "margin_sweep_pareto.csv", index=False)

    payload = {
        "output_dir": str(outdir.resolve()),
        "config": {
            "test_size": args.test_size,
            "calib_size": args.calib_size,
            "prediction_margin_threshold": args.prediction_margin,
            "margin_grid": {
                "min": args.margin_min,
                "max": args.margin_max,
                "step": args.margin_step,
            },
            "joint_score": {
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
            },
            "n_lags": args.n_lags,
            "tcn_device": args.tcn_device,
        },
        "files_written": [
            "per_company_metrics.csv",
            "metrics_summary.csv",
            "margin_sweep.csv",
            "margin_sweep_summary.csv",
            "margin_sweep_best.csv",
            "margin_sweep_pareto.csv",
        ],
    }
    with open(outdir / "run_summary.json", "w") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
