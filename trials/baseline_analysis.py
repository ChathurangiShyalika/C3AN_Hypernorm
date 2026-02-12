from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
    df = df.sort_values(["company", "date"])
    return df


def evaluate_baseline_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    company: str,
) -> Dict[str, float]:
    """Train linear regression and compute evaluation metrics."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "train_mape": np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100,
        "test_mape": np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100,
    }
    
    # Store coefficients
    metrics["coef_open"] = model.coef_[0]
    metrics["coef_high"] = model.coef_[1]
    metrics["coef_low"] = model.coef_[2]
    metrics["intercept"] = model.intercept_
    
    return metrics, model, y_pred_test


def run_baseline_analysis(df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """Run baseline linear regression for each company."""
    results = []
    models = {}
    
    for company, group in df.groupby("company", observed=True):
        if len(group) < 50:  # Skip companies with insufficient data
            continue
        
        X = group[["open", "high", "low"]].values
        y = group["close"].values
        
        # Split chronologically (no shuffle to preserve time order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        metrics, model, y_pred = evaluate_baseline_model(
            X_train, X_test, y_train, y_test, company
        )
        
        metrics["company"] = company
        metrics["n_samples"] = len(group)
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        
        results.append(metrics)
        models[company] = model
    
    results_df = pd.DataFrame(results)
    return results_df, models


def plot_baseline_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create visualization of baseline model performance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R2 scores
    ax = axes[0, 0]
    results_sorted = results_df.sort_values("test_r2", ascending=False).head(15)
    x = range(len(results_sorted))
    ax.barh(x, results_sorted["train_r2"], alpha=0.6, label="Train R²")
    ax.barh(x, results_sorted["test_r2"], alpha=0.8, label="Test R²")
    ax.set_yticks(x)
    ax.set_yticklabels(results_sorted["company"], fontsize=8)
    ax.set_xlabel("R² Score")
    ax.set_title("R² Scores (Top 15 Companies)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAPE
    ax = axes[0, 1]
    results_sorted_mape = results_df.sort_values("test_mape").head(15)
    x = range(len(results_sorted_mape))
    ax.barh(x, results_sorted_mape["train_mape"], alpha=0.6, label="Train MAPE")
    ax.barh(x, results_sorted_mape["test_mape"], alpha=0.8, label="Test MAPE")
    ax.set_yticks(x)
    ax.set_yticklabels(results_sorted_mape["company"], fontsize=8)
    ax.set_xlabel("MAPE (%)")
    ax.set_title("Mean Absolute Percentage Error (Top 15)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Coefficient distribution
    ax = axes[1, 0]
    coef_data = results_df[["coef_open", "coef_high", "coef_low"]].values
    ax.boxplot(coef_data, labels=["Open", "High", "Low"])
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Feature Coefficient Distribution")
    ax.grid(True, alpha=0.3)
    
    # Train vs Test R2 scatter
    ax = axes[1, 1]
    ax.scatter(results_df["train_r2"], results_df["test_r2"], alpha=0.6)
    max_val = max(results_df["train_r2"].max(), results_df["test_r2"].max())
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Perfect Fit")
    ax.set_xlabel("Train R²")
    ax.set_ylabel("Test R²")
    ax.set_title("Train vs Test R² (Overfitting Check)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def analyze_feature_importance(results_df: pd.DataFrame) -> None:
    """Print analysis of which features are most important."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    avg_coefs = {
        "Open": results_df["coef_open"].mean(),
        "High": results_df["coef_high"].mean(),
        "Low": results_df["coef_low"].mean(),
    }
    
    print("\nAverage coefficients across all companies:")
    for feature, coef in sorted(avg_coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feature:10s}: {coef:8.4f}")
    
    print(f"\nAverage intercept: {results_df['intercept'].mean():.4f}")
    
    # Check if the model is essentially just averaging the features
    total_coef = sum(abs(c) for c in avg_coefs.values())
    print(f"\nSum of absolute coefficients: {total_coef:.4f}")
    print("(Values close to 1.0 suggest the model is mostly averaging the features)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline linear regression analysis for stock close price prediction."
    )
    parser.add_argument("--asset-file", type=Path, default=Path("asset_events_df.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data for testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("Loading data...")
    df = load_and_prepare_data(args.asset_file)
    print(f"Loaded {len(df)} records for {df['company'].nunique()} companies")
    
    print("\nRunning baseline linear regression analysis...")
    results_df, models = run_baseline_analysis(df, test_size=args.test_size)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_dir / "baseline_results.csv", index=False)
    
    plot_baseline_results(results_df, args.output_dir / "baseline_performance.png")
    
    analyze_feature_importance(results_df)    
    print(f"\nOutputs saved to {args.output_dir}/")
    print("  - baseline_results.csv")
    print("  - baseline_performance.png")


if __name__ == "__main__":
    main()
