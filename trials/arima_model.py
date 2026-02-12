import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from model_utils import load_and_prepare_data, WalkForwardValidator, calculate_metrics, reconstruct_price_forecasts, debug_pred_true
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ASSET_PATH = Path("asset_events_df.csv")
HORIZON = 10
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_arima_analysis():
    print("Loading data...")
    df = load_and_prepare_data(ASSET_PATH)
    companies = df['company'].unique()
    
    results = []
    
    print(f"Starting ARIMA analysis for {len(companies)} companies...")
    
    for i, company in enumerate(companies):
        if i % 5 == 0:
            print(f"Processing company {i+1}/{len(companies)}: {company}")
            
        company_df = df[df['company'] == company].sort_values('date').reset_index(drop=True)
    
        company_df['log_ret'] = np.log(company_df['close'] / company_df['close'].shift(1))
    
        for h in range(1, HORIZON + 1):
            company_df[f'target_close_h{h}'] = company_df['close'].shift(-h)
        target_cols = [f"target_close_h{h}" for h in range(1, HORIZON+1)]
        model_data = company_df.dropna(subset=['log_ret'] + target_cols).reset_index(drop=True)
        
        # VALIDATION SANITY CHECKS
        zero_std_frac = (model_data[target_cols].std(axis=1) == 0).mean()
        print(f"Sanity: fraction zero-std across horizon targets for {company} = {zero_std_frac:.3%}")
        
        if zero_std_frac > 0.01:
            print(f"WARNING: High fraction of zero-std targets for {company}. Inspecting first 5 problematic rows:")
            problem_rows = model_data[model_data[target_cols].std(axis=1) == 0].head(5)
            print(problem_rows[['date', 'close'] + target_cols])
        
        if len(model_data) < 50:
            continue
            
        series = model_data['log_ret']
        prices = model_data['close']
        dates = model_data['date']
        
        validator = WalkForwardValidator(n_splits=5)
        
        for split_idx, (train_idx, test_idx) in enumerate(validator.split(series)):
            train_ret = series.iloc[train_idx]
            test_ret = series.iloc[test_idx]
            
            if len(test_ret) < HORIZON:
                continue

            try:
                model = ARIMA(train_ret.values, order=(5,0,1))
                model_fit = model.fit()
                
                # Forecast Returns
                pred_ret = model_fit.forecast(steps=HORIZON)
                
                last_train_price = prices.iloc[train_idx[-1]]
                
                pred_prices = reconstruct_price_forecasts(last_train_price, pred_ret, return_type='log')

                origin_idx = train_idx[-1]
                target_cols = [f'target_close_h{h}' for h in range(1, HORIZON + 1)]
                
                actual_prices = model_data.iloc[origin_idx][target_cols].values.astype(float)
                
                debug_pred_true(actual_prices, pred_prices, name=f"ARIMA_{company}_{split_idx}")
                metrics = calculate_metrics(actual_prices, pred_prices)
                metrics['company'] = company
                metrics['split'] = split_idx
                results.append(metrics)
                
            except Exception as e:
                # print(f"ARIMA Error {company}: {e}")
                pass
                
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_path = OUTPUT_DIR / "arima_results.csv"
        results_df.to_csv(output_path, index=False)
        
        company_summary = (
            results_df.groupby('company')[['mae', 'rmse', 'mape', 'r2']]
            .mean()
            .reset_index()
            .sort_values('rmse', ascending=False)
        )
        summary_path = OUTPUT_DIR / "arima_company_summary.csv"
        company_summary.to_csv(summary_path, index=False)

        avg_metrics = results_df[['mae', 'rmse', 'mape', 'r2']].mean()
        print("\nAverage ARIMA Performance (on Prices):")
        print(avg_metrics)
        print("\nPer-company summary (worst RMSE first):")
        print(company_summary)
        print(f"\nDetailed results saved to {output_path}")
        print(f"Company summary saved to {summary_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_arima_analysis()
