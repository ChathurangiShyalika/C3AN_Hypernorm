from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from direction_pipeline import run_direction_pipeline

DELTA_VALUES = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]
OUTPUT_ROOT = Path("analysis_outputs/operability_sweep")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = dict(
    horizons=(1, 2, 3, 4, 5),
    n_lags=10,
    flat_threshold_strategy="atr_sqrt",
    flat_threshold_pct=0.1,
    atr_threshold_multiplier=0.5,
    atr_up_multiplier=0.6,
    atr_down_multiplier=0.5,
    use_technical_indicators=True,
    aggregate_daily=True,
    model_type="ensemble",
    ensemble_logistic_weight=0.5,
    balance_classes=True,
    test_size=0.2,
    n_splits_cv=5,
)


def _compute_directional_recalls(preds: pd.DataFrame) -> dict:
    recalls = {}
    for label in ("down", "up"):
        mask = preds["true_label"] == label
        recalls[f"recall_{label}"] = float(
            (preds.loc[mask, "pred_label"] == label).mean()
        ) if mask.any() else np.nan
    directional_mask = preds["true_label"].isin(["down", "up"])
    recalls["recall_directional_avg"] = float(
        (preds.loc[directional_mask, "pred_label"] == preds.loc[directional_mask, "true_label"]).mean()
    ) if directional_mask.any() else np.nan
    return recalls


def _compute_margin_stats(margins: pd.Series, quantiles: List[float]) -> dict:
    margins = margins.dropna()
    if margins.empty:
        return {f"margin_p{int(q*100)}": np.nan for q in quantiles}
    q_values = margins.quantile(quantiles)
    return {f"margin_p{int(q*100)}": float(q_values.loc[q]) for q in quantiles}


def main():
    data_file = Path("asset_events_df.csv")
    df = pd.read_csv(data_file, parse_dates=["date"])
    df["company"] = df["company"].astype("category")

    summary_rows = []
    margin_rows = []
    quantiles = [0.5, 0.75, 0.9, 0.95]

    for delta in DELTA_VALUES:
        delta_tag = f"delta_{delta:.2f}".replace(".", "p")
        outdir = OUTPUT_ROOT / delta_tag
        run_direction_pipeline(
            df.copy(),
            outdir,
            prediction_margin_threshold=delta,
            log_run_metadata=True,
            **BASE_CONFIG,
        )

        for horizon in BASE_CONFIG["horizons"]:
            preds_path = outdir / f"direction_preds_h{horizon}.csv"
            if not preds_path.exists():
                continue
            preds = pd.read_csv(preds_path, parse_dates=["base_date", "target_date"])
            operable_mask = preds["operable"].astype(bool)
            operable_share = float(operable_mask.mean())
            operable_accuracy = float(
                preds.loc[operable_mask, "is_correct"].mean()
            ) if operable_mask.any() else np.nan

            recalls = _compute_directional_recalls(preds)
            margin_stats = _compute_margin_stats(preds["proba_margin"], quantiles)

            summary_rows.append({
                "delta": float(delta),
                "horizon": int(horizon),
                "operable_share": operable_share,
                "operable_accuracy": operable_accuracy,
                **recalls,
            })
            margin_rows.append({
                "delta": float(delta),
                "horizon": int(horizon),
                **margin_stats,
            })

    summary_df = pd.DataFrame(summary_rows)
    margin_df = pd.DataFrame(margin_rows)
    summary_df.to_csv(OUTPUT_ROOT / "operability_sweep_summary.csv", index=False)
    margin_df.to_csv(OUTPUT_ROOT / "operability_margin_distribution.csv", index=False)

    print("Sweep complete. Results stored in:")
    print(OUTPUT_ROOT / "operability_sweep_summary.csv")
    print(OUTPUT_ROOT / "operability_margin_distribution.csv")


if __name__ == "__main__":
    main()
