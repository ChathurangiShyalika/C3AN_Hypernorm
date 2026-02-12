import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from model_utils import load_and_prepare_data, WalkForwardValidator, calculate_metrics, make_lagged_dataset
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ASSET_PATH = Path("asset_events_df.csv")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_linear_baseline():
    print("Running Linear Baseline...")
    df = load_and_prepare_data(ASSET_PATH)
    companies = df['company'].unique()
    results = []
    
    for i, company in enumerate(companies):
        company_df = df[df['company'] == company].sort_values('date').reset_index(drop=True)
        lagged_df = make_lagged_dataset(company_df, n_lags=10, horizon=5)
        
        if len(lagged_df) < 50:
            continue
            
        feature_cols = [c for c in lagged_df.columns if c.startswith('lag_')]
        target_cols = [c for c in lagged_df.columns if c.startswith('target_') and not c.startswith('target_date')]
        
        X = lagged_df[feature_cols].values
        y = lagged_df[target_cols].values
        
        validator = WalkForwardValidator(n_splits=5)
        
        for split_idx, (train_idx, test_idx) in enumerate(validator.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = MultiOutputRegressor(LinearRegression())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = calculate_metrics(y_test, y_pred)
            metrics['company'] = company
            metrics['split'] = split_idx
            metrics['model'] = 'Linear Baseline'
            results.append(metrics)
            
    return pd.DataFrame(results)

def load_results(filename, model_name):
    path = OUTPUT_DIR / filename
    if path.exists():
        df = pd.read_csv(path)
        df['model'] = model_name
        return df
    else:
        print(f"Warning: {filename} not found.")
        return pd.DataFrame()

def compare_models():
    # Run Baseline
    baseline_results = run_linear_baseline()
    
    # Load other results
    arima_results = load_results("arima_results.csv", "ARIMA")
    xgboost_results = load_results("xgboost_results.csv", "XGBoost")
    
    # Combine
    all_results = pd.concat([baseline_results, arima_results, xgboost_results], ignore_index=True)
    
    if all_results.empty:
        print("No results to compare.")
        return

    # Save combined results
    all_results.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
    
    # Aggregate metrics
    summary = all_results.groupby('model')[['mae', 'rmse', 'mape', 'r2']].mean().reset_index()
    print("\nModel Comparison Summary:")
    print(summary)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # MAE Comparison
    plt.subplot(1, 2, 1)
    sns.barplot(data=all_results, x='model', y='mae', errorbar='sd')
    plt.title('MAE Comparison (Lower is Better)')
    plt.ylabel('MAE')
    
    # R2 Comparison
    plt.subplot(1, 2, 2)
    sns.barplot(data=all_results, x='model', y='r2', errorbar='sd')
    plt.title('R2 Comparison (Higher is Better)')
    plt.ylabel('R2')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png")
    print(f"\nComparison plot saved to {OUTPUT_DIR / 'model_comparison.png'}")

if __name__ == "__main__":
    compare_models()
