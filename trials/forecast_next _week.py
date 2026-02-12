from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit

HORIZON = 5 
N_LAGS = 10  
DIRECTION_THRESHOLD = 0.002 
PRICE_FEATURE_COL = "ohlc_avg"
DIRECTION_ORDER = ["down", "flat", "up"]


def load_and_prepare_data(asset_path: Path) -> pd.DataFrame:
    """Load asset data and ensure numeric columns are properly typed."""
    df = pd.read_csv(
        asset_path,
        parse_dates=["date"],
        dtype={"company": "category"},
        low_memory=False,
    )
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values in the key columns
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Aggregate to one observation per company/date to avoid duplicate-day leakage
    aggregated = (
        df.groupby(["company", "date"], observed=True)
        .agg(
            open=("open", "mean"),
            high=("high", "mean"),
            low=("low", "mean"),
            close=("close", "mean"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )

    aggregated[PRICE_FEATURE_COL] = aggregated[["open", "high", "low", "close"]].mean(axis=1)
    aggregated["company"] = aggregated["company"].astype("category")
    aggregated = aggregated.sort_values(["company", "date"]).reset_index(drop=True)
    return aggregated


def summarize_data(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate descriptive statistics to understand the dataset before modeling."""
    output_dir.mkdir(parents=True, exist_ok=True)

    company_summary = (
        df.groupby("company", observed=True)
        .agg(
            n_rows=("close", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            mean_close=("close", "mean"),
            std_close=("close", "std"),
            mean_volume=("volume", "mean"),
            price_avg_mean=(PRICE_FEATURE_COL, "mean"),
            price_avg_std=(PRICE_FEATURE_COL, "std"),
        )
        .reset_index()
    )
    company_summary.to_csv(output_dir / "data_summary_per_company.csv", index=False)

    overall_stats = df[["open", "high", "low", "close", PRICE_FEATURE_COL]].describe()
    overall_stats.to_csv(output_dir / "data_summary_overall.csv")

    corr_matrix = df[["open", "high", "low", "close", PRICE_FEATURE_COL]].corr()
    corr_matrix.to_csv(output_dir / "data_correlation_matrix.csv")

    print("Data summaries saved:")
    print("  - data_summary_per_company.csv")
    print("  - data_summary_overall.csv")
    print("  - data_correlation_matrix.csv")


def _direction_count_table(df: pd.DataFrame, direction_col: str) -> pd.DataFrame:
    counts = (
        df.groupby("company", observed=True)[direction_col]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=DIRECTION_ORDER, fill_value=0)
        .reset_index()
    )
    counts["total_samples"] = counts[DIRECTION_ORDER].sum(axis=1)
    for label in DIRECTION_ORDER:
        counts[f"{label}_pct"] = np.where(
            counts["total_samples"] > 0,
            counts[label] / counts["total_samples"],
            0.0,
        )
    return counts


def compute_pre_analysis_direction_counts(
    df: pd.DataFrame, output_dir: Path, threshold: float = DIRECTION_THRESHOLD
) -> pd.DataFrame:
    df_sorted = df.sort_values(["company", "date"])
    df_sorted["ret"] = df_sorted.groupby("company")[PRICE_FEATURE_COL].pct_change()
    df_sorted = df_sorted.dropna(subset=["ret"])
    df_sorted["direction"] = df_sorted["ret"].apply(lambda r: _label_direction(r, threshold))

    pre_counts = _direction_count_table(df_sorted, "direction")
    pre_counts.to_csv(output_dir / "direction_counts_pre_company.csv", index=False)
    print("  - direction_counts_pre_company.csv")
    return pre_counts


def make_lagged_dataset(
    group: pd.DataFrame,
    n_lags: int = N_LAGS,
    horizon: int = HORIZON,
    feature_col: str = PRICE_FEATURE_COL,
) -> pd.DataFrame:
    """
    For a single company group (sorted by date), create lag features of close and
    multi-step targets for the next `horizon` closes.
    Returns a DataFrame with feature columns and target columns:
      - lag_1, lag_2, ..., lag_n_lags  (lag_1 = close_t, lag_2 = close_{t-1}, ...)
      - target_1, target_2, ..., target_horizon (target_1 = close_{t+1}, ...)
    """
    g = group.copy().reset_index(drop=True)

    for lag in range(1, n_lags + 1):
        g[f"lag_{lag}"] = g[feature_col].shift(lag - 1)


    for h in range(1, horizon + 1):
        g[f"target_{h}"] = g["close"].shift(-h)  # target_1 = close_{t+1}
        g[f"target_date_{h}"] = g["date"].shift(-h)


    cols_to_check = [f"lag_{i}" for i in range(1, n_lags + 1)] + [
        f"target_{h}" for h in range(1, horizon + 1)
    ]
    g = g.dropna(subset=cols_to_check).reset_index(drop=True)
    return g


def evaluate_forecast_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    company: str,
) -> Tuple[Dict[str, float], MultiOutputRegressor, np.ndarray]:
    """
    Train a multi-output regressor and compute metrics for each horizon.
    Returns: (metrics_dict, trained_model, y_pred_test)
    """
    base = LinearRegression()
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    horizon = y_test.shape[1]
    train_mae_per_h = []
    test_mae_per_h = []
    train_rmse_per_h = []
    test_rmse_per_h = []
    train_mape_per_h = []
    test_mape_per_h = []
    train_r2_per_h = []
    test_r2_per_h = []

    for h in range(horizon):
        y_tr = y_train[:, h]
        y_tr_pred = y_pred_train[:, h]
        y_te = y_test[:, h]
        y_te_pred = y_pred_test[:, h]

        train_mae_per_h.append(mean_absolute_error(y_tr, y_tr_pred))
        test_mae_per_h.append(mean_absolute_error(y_te, y_te_pred))

        train_rmse_per_h.append(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
        test_rmse_per_h.append(np.sqrt(mean_squared_error(y_te, y_te_pred)))

      
        train_mape_per_h.append(mean_absolute_percentage_error(y_tr, y_tr_pred) * 100)
        test_mape_per_h.append(mean_absolute_percentage_error(y_te, y_te_pred) * 100)

        train_r2_per_h.append(r2_score(y_tr, y_tr_pred))
        test_r2_per_h.append(r2_score(y_te, y_te_pred))


    metrics = {
        "company": company,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "train_mae_per_h": train_mae_per_h,
        "test_mae_per_h": test_mae_per_h,
        "train_rmse_per_h": train_rmse_per_h,
        "test_rmse_per_h": test_rmse_per_h,
        "train_mape_per_h": train_mape_per_h,
        "test_mape_per_h": test_mape_per_h,
        "train_r2_per_h": train_r2_per_h,
        "test_r2_per_h": test_r2_per_h,
        "train_mae_mean": float(np.mean(train_mae_per_h)),
        "test_mae_mean": float(np.mean(test_mae_per_h)),
        "train_rmse_mean": float(np.mean(train_rmse_per_h)),
        "test_rmse_mean": float(np.mean(test_rmse_per_h)),
        "train_mape_mean": float(np.mean(train_mape_per_h)),
        "test_mape_mean": float(np.mean(test_mape_per_h)),
        "train_r2_mean": float(np.mean(train_r2_per_h)),
        "test_r2_mean": float(np.mean(test_r2_per_h)),
    }

    return metrics, model, y_pred_test


def compute_out_of_bag_errors(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Approximate out-of-bag error via rolling TimeSeriesSplit cross-validation."""
    if len(X) <= n_splits + 1:
        return {}

    n_splits = min(n_splits, max(2, len(X) - 1))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, test_idx in splitter.split(X):
        if len(test_idx) == 0:
            continue
        fold_model = MultiOutputRegressor(LinearRegression())
        fold_model.fit(X[train_idx], y[train_idx])
        preds = fold_model.predict(X[test_idx])

        fold_metrics.append(
            {
                "mae": mean_absolute_error(y[test_idx], preds),
                "rmse": np.sqrt(mean_squared_error(y[test_idx], preds)),
                "mape": mean_absolute_percentage_error(y[test_idx], preds) * 100,
                "r2": r2_score(y[test_idx], preds, multioutput="variance_weighted"),
            }
        )

    if not fold_metrics:
        return {}

    return {
        "oob_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "oob_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "oob_mape": float(np.mean([m["mape"] for m in fold_metrics])),
        "oob_r2": float(np.mean([m["r2"] for m in fold_metrics])),
    }

def run_forecast_analysis(
    df: pd.DataFrame, test_size: float = 0.2, n_lags: int = N_LAGS, horizon: int = HORIZON
) -> Tuple[pd.DataFrame, Dict[str, MultiOutputRegressor], pd.DataFrame]:
    """Run multi-step forecasting baseline for each company."""
    results = []
    models = {}
    all_predictions: List[Dict[str, object]] = []
    for company, group in df.groupby("company", observed=True):
        if len(group) < (n_lags + horizon + 10):  

            continue
        g = make_lagged_dataset(group, n_lags=n_lags, horizon=horizon)
        if g.empty:
            continue

        lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
        target_cols = [f"target_{h}" for h in range(1, horizon + 1)]

        X = g[lag_cols].values  
        y = g[target_cols].values 


        split_idx = int(len(X) * (1 - test_size))
        if split_idx < 1 or (len(X) - split_idx) < 1:

            continue

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        metrics, model, y_pred = evaluate_forecast_model(
            X_train, X_test, y_train, y_test, company
        )

        metrics["n_samples"] = int(len(g))
        metrics["company"] = company

        oob_metrics = compute_out_of_bag_errors(X, y)
        metrics.update(oob_metrics)

        results.append(metrics)
        models[company] = model


        for idx, global_idx in enumerate(range(split_idx, len(g))):
            base_row = g.loc[global_idx]
            base_close = float(base_row["close"])
            base_date = base_row["date"]
            for h in range(1, horizon + 1):
                target_date = base_row.get(f"target_date_{h}")
                all_predictions.append(
                    {
                        "company": company,
                        "base_date": base_date,
                        "base_close": base_close,
                        "target_date": target_date,
                        "horizon": h,
                        "actual_close": float(y_test[idx, h - 1]),
                        "predicted_close": float(y_pred[idx, h - 1]),
                    }
                )


    if results:
        results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame(
            columns=[
                "company",
                "n_samples",
                "n_train",
                "n_test",
                "train_mae_mean",
                "test_mae_mean",
                "train_rmse_mean",
                "test_rmse_mean",
                "train_mape_mean",
                "test_mape_mean",
                "train_r2_mean",
                "test_r2_mean",
            ]
        )
    predictions_df = pd.DataFrame(all_predictions)
    return results_df, models, predictions_df


def plot_forecast_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create visualization of forecast performance (aggregated across horizons)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if results_df.empty:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))


    ax = axes[0, 0]
    results_sorted = results_df.sort_values("test_mae_mean", ascending=True).head(15)
    x = range(len(results_sorted))
    ax.barh(x, results_sorted["train_mae_mean"], alpha=0.6, label="Train MAE")
    ax.barh(x, results_sorted["test_mae_mean"], alpha=0.8, label="Test MAE")
    ax.set_yticks(x)
    ax.set_yticklabels(results_sorted["company"], fontsize=8)
    ax.set_xlabel("MAE (mean across horizons)")
    ax.set_title("Mean Absolute Error (Top 15 Companies)")
    ax.legend()
    ax.grid(True, alpha=0.3)


    ax = axes[0, 1]
    results_sorted_mape = results_df.sort_values("test_mape_mean", ascending=True).head(15)
    x = range(len(results_sorted_mape))
    ax.barh(x, results_sorted_mape["train_mape_mean"], alpha=0.6, label="Train MAPE")
    ax.barh(x, results_sorted_mape["test_mape_mean"], alpha=0.8, label="Test MAPE")
    ax.set_yticks(x)
    ax.set_yticklabels(results_sorted_mape["company"], fontsize=8)
    ax.set_xlabel("MAPE (%) (mean across horizons)")
    ax.set_title("Mean Absolute Percentage Error (Top 15)")
    ax.legend()
    ax.grid(True, alpha=0.3)


    ax = axes[1, 0]
    ax.scatter(results_df["train_r2_mean"], results_df["test_r2_mean"], alpha=0.6)
    max_val = max(results_df["train_r2_mean"].max(), results_df["test_r2_mean"].max())
    ax.plot([min(0, -1), max_val], [min(0, -1), max_val], "r--", alpha=0.5, label="Perfect Fit")
    ax.set_xlabel("Train R² (mean across horizons)")
    ax.set_ylabel("Test R² (mean across horizons)")
    ax.set_title("Train vs Test R² (Overfitting Check)")
    ax.legend()
    ax.grid(True, alpha=0.3)


    ax = axes[1, 1]
    ax.boxplot(results_df["test_mae_mean"].dropna())
    ax.set_title("Distribution of Test MAE (mean across horizons)")
    ax.set_ylabel("MAE")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def analyze_feature_importance_dummy(results_df: pd.DataFrame) -> None:
    """
    For lag-based features, a simple interpretation is less direct.
    Here we'll just print summary metrics across companies (mean of means).
    """
    print("\n" + "=" * 60)
    print("FORECAST BASELINE SUMMARY (aggregated across companies & horizons)")
    print("=" * 60)
    if results_df.empty:
        print("No results available.")
        return

    print(f"Average test MAE (mean across horizons & companies): {results_df['test_mae_mean'].mean():.4f}")
    print(f"Average test RMSE (mean across horizons & companies): {results_df['test_rmse_mean'].mean():.4f}")
    print(f"Average test MAPE (%): {results_df['test_mape_mean'].mean():.4f}")
    print(f"Average test R2 (mean): {results_df['test_r2_mean'].mean():.4f}")


def _label_direction(ret: float, threshold: float) -> str:
    if ret > threshold:
        return "up"
    if ret < -threshold:
        return "down"
    return "flat"


def compute_directional_accuracy(
    predictions_df: pd.DataFrame, threshold: float = DIRECTION_THRESHOLD
) -> Dict[str, pd.DataFrame | float]:
    """Compute directional accuracy metrics from prediction records."""
    if predictions_df.empty:
        return {}

    df = predictions_df.copy()
    if "base_close" not in df:
        raise ValueError("predictions_df must contain 'base_close' column for direction computation")

    df["actual_ret"] = (df["actual_close"] - df["base_close"]) / df["base_close"]
    df["predicted_ret"] = (df["predicted_close"] - df["base_close"]) / df["base_close"]

    df["actual_dir"] = df["actual_ret"].apply(lambda r: _label_direction(r, threshold))
    df["predicted_dir"] = df["predicted_ret"].apply(lambda r: _label_direction(r, threshold))
    df["hit"] = (df["actual_dir"] == df["predicted_dir"]).astype(float)

    overall_accuracy = df["hit"].mean()
    per_horizon = df.groupby("horizon")["hit"].mean().reset_index(name="directional_accuracy")
    per_company = df.groupby("company")["hit"].mean().reset_index(name="directional_accuracy")
    per_company_horizon = (
        df.groupby(["company", "horizon"])["hit"].mean().reset_index(name="directional_accuracy")
    )

    return {
        "overall_accuracy": float(overall_accuracy),
        "per_horizon": per_horizon,
        "per_company": per_company,
        "per_company_horizon": per_company_horizon,
        "detailed": df,
    }


def generate_confusion_tables(direction_metrics: Dict[str, pd.DataFrame | float]) -> Dict[str, pd.DataFrame]:
    detailed = direction_metrics.get("detailed")
    if detailed is None or isinstance(detailed, float) or detailed.empty:
        return {}

    actual_pred = pd.crosstab(
        detailed["actual_dir"], detailed["predicted_dir"], dropna=False
    )
    normalized = pd.crosstab(
        detailed["actual_dir"], detailed["predicted_dir"], dropna=False, normalize="index"
    )

    return {
        "confusion_counts": actual_pred.reset_index(),
        "confusion_normalized": normalized.reset_index(),
    }


def compute_post_analysis_direction_counts(
    direction_metrics: Dict[str, pd.DataFrame | float], output_dir: Path
) -> Dict[str, pd.DataFrame]:
    detailed = direction_metrics.get("detailed")
    if detailed is None or isinstance(detailed, float) or detailed.empty:
        return {}

    actual_counts = _direction_count_table(detailed, "actual_dir")
    predicted_counts = _direction_count_table(detailed, "predicted_dir")

    actual_counts.to_csv(output_dir / "direction_counts_post_actual.csv", index=False)
    predicted_counts.to_csv(output_dir / "direction_counts_post_predicted.csv", index=False)

    print("  - direction_counts_post_actual.csv")
    print("  - direction_counts_post_predicted.csv")

    return {
        "post_actual": actual_counts,
        "post_predicted": predicted_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-step (next-week) linear regression forecasting using lag features."
    )
    parser.add_argument("--asset-file", type=Path, default=Path("asset_events_df.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data for testing")
    parser.add_argument("--n-lags", type=int, default=N_LAGS, help="Number of lagged closes to use as features")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Forecast horizon in days (default 5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading data...")
    df = load_and_prepare_data(args.asset_file)
    print(f"Loaded {len(df)} records for {df['company'].nunique()} companies")

    print("\nRunning exploratory data analysis summaries...")
    summarize_data(df, args.output_dir)
    compute_pre_analysis_direction_counts(df, args.output_dir)

    print("\nRunning multi-step (next-week) forecasting baseline...")
    results_df, models, predictions_df = run_forecast_analysis(
        df, test_size=args.test_size, n_lags=args.n_lags, horizon=args.horizon
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_dir / "forecast_baseline_results.csv", index=False)
    predictions_df.to_csv(args.output_dir / "forecast_baseline_predictions.csv", index=False)

    plot_forecast_results(results_df, args.output_dir / "forecast_baseline_performance.png")

    analyze_feature_importance_dummy(results_df)

    direction_metrics = compute_directional_accuracy(predictions_df)
    if direction_metrics:
        overall_pct = direction_metrics["overall_accuracy"] * 100
        print(f"\nDirectional accuracy (overall): {overall_pct:.2f}%")

        direction_metrics["per_horizon"].to_csv(
            args.output_dir / "directional_accuracy_per_horizon.csv", index=False
        )
        direction_metrics["per_company"].to_csv(
            args.output_dir / "directional_accuracy_per_company.csv", index=False
        )
        direction_metrics["per_company_horizon"].to_csv(
            args.output_dir / "directional_accuracy_per_company_horizon.csv", index=False
        )
        direction_metrics["detailed"].to_csv(
            args.output_dir / "directional_accuracy_detailed.csv", index=False
        )

        confusion_tables = generate_confusion_tables(direction_metrics)
        if confusion_tables:
            confusion_tables["confusion_counts"].to_csv(
                args.output_dir / "direction_confusion_counts.csv", index=False
            )
            confusion_tables["confusion_normalized"].to_csv(
                args.output_dir / "direction_confusion_normalized.csv", index=False
            )

        compute_post_analysis_direction_counts(direction_metrics, args.output_dir)

        print("Directional accuracy files saved:")
        print("  - directional_accuracy_per_horizon.csv")
        print("  - directional_accuracy_per_company.csv")
        print("  - directional_accuracy_per_company_horizon.csv")
        print("  - directional_accuracy_detailed.csv")
        print("  - direction_confusion_counts.csv")
        print("  - direction_confusion_normalized.csv")

    print(f"\nOutputs saved to {args.output_dir}/")
    print("  - data_summary_per_company.csv")
    print("  - data_summary_overall.csv")
    print("  - data_correlation_matrix.csv")
    print("  - direction_counts_pre_company.csv")
    print("  - forecast_baseline_results.csv")
    print("  - forecast_baseline_predictions.csv")
    print("  - forecast_baseline_performance.png")


if __name__ == "__main__":
    main()
