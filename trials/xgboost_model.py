import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import HuberRegressor
from model_utils import (
    load_and_prepare_data,
    WalkForwardValidator,
    calculate_metrics,
    create_price_feature_frame,
)
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ASSET_PATH = Path("asset_events_df.csv")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
HORIZON = 5
VOL_SCALE_MIN = 1e-3
VOL_SCALE_MAX = 0.5
VAL_FRACTION = 0.15
MIN_VAL_SAMPLES = 40
HUBER_BLEND = 0.3
LOG_RET_CLIP = 0.4

def run_xgboost_analysis():
    print("Loading data...")
    df = load_and_prepare_data(ASSET_PATH)
    companies = df['company'].unique()
    results = []

    print(f"Starting XGBoost analysis for {len(companies)} companies...")

    for i, company in enumerate(companies):
        if i % 5 == 0:
            print(f"Processing company {i+1}/{len(companies)}: {company}")

        company_df = df[df['company'] == company].sort_values('date').reset_index(drop=True)
        model_data, feature_cols, target_cols = create_price_feature_frame(
            company_df,
            horizon=HORIZON,
            n_price_lags=20,
            n_return_lags=15,
        )

        if len(model_data) < 50:
            continue

        X = model_data[feature_cols].values
        target_price_matrix = model_data[target_cols].values
        ref_prices = model_data['close'].values

        log_returns = np.log(target_price_matrix) - np.log(ref_prices[:, None])
        vol_cols = [c for c in feature_cols if c.startswith('roll_ret_std_')]
        if vol_cols:
            vol_matrix = model_data[vol_cols].values
            vol_scale_raw = np.nanmean(vol_matrix, axis=1)
        else:
            vol_scale_raw = model_data['return_1'].rolling(5).std().fillna(0).values

        vol_scale_raw = np.nan_to_num(vol_scale_raw, nan=0.0, posinf=0.0, neginf=0.0)
        vol_scale = np.clip(vol_scale_raw, VOL_SCALE_MIN, VOL_SCALE_MAX)

        y = log_returns / vol_scale[:, None]

        validator = WalkForwardValidator(n_splits=5)

        for split_idx, (train_idx, test_idx) in enumerate(validator.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            n_train_fold = len(X_train)
            val_size = max(MIN_VAL_SAMPLES, int(VAL_FRACTION * n_train_fold))
            if val_size >= n_train_fold - 20:
                val_size = max(10, n_train_fold // 5)
            train_inner_end = n_train_fold - val_size
            if train_inner_end < 30:
                train_inner_end = n_train_fold
                val_size = 0

            X_inner = X_train[:train_inner_end]
            y_inner = y_train[:train_inner_end]
            if val_size > 0:
                X_val = X_train[train_inner_end:]
                y_val = y_train[train_inner_end:]
            else:
                X_val = None
                y_val = None

            horizon_preds = np.zeros_like(y_test)

            for horizon_idx in range(y.shape[1]):
                if X_val is not None and len(X_val) > 0:
                    dtrain = xgb.DMatrix(X_inner, label=y_inner[:, horizon_idx])
                    dval = xgb.DMatrix(X_val, label=y_val[:, horizon_idx])
                    dtest = xgb.DMatrix(X_test)
                    params = {
                        'eta': 0.04,
                        'max_depth': 6,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'gamma': 0.15,
                        'lambda': 1.5,
                        'alpha': 0.2,
                        'objective': 'reg:squarederror',
                        'eval_metric': 'mae',
                        'verbosity': 0,
                        'nthread': 4,
                    }
                    booster = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=1200,
                        evals=[(dval, 'validation')],
                        early_stopping_rounds=60,
                        verbose_eval=False,
                    )
                    best_ntree = booster.best_iteration + 1 if booster.best_iteration is not None else booster.best_ntree_limit
                    xgb_pred = booster.predict(dtest, iteration_range=(0, best_ntree))
                else:
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=600,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        gamma=0.15,
                        reg_lambda=1.5,
                        reg_alpha=0.2,
                        objective='reg:squarederror',
                        n_jobs=4,
                        random_state=42,
                    )
                    xgb_model.fit(X_inner, y_inner[:, horizon_idx])
                    xgb_pred = xgb_model.predict(X_test)

                try:
                    huber_model = HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=500)
                    huber_model.fit(X_inner, y_inner[:, horizon_idx])
                    huber_pred = huber_model.predict(X_test)
                    combined_pred = (1 - HUBER_BLEND) * xgb_pred + HUBER_BLEND * huber_pred
                except Exception:
                    combined_pred = xgb_pred

                horizon_preds[:, horizon_idx] = combined_pred

            test_scales = vol_scale[test_idx]

            for j in range(len(test_idx)):
                idx = test_idx[j]
                actual_prices = target_price_matrix[idx]
                scaled_logs = horizon_preds[j] * test_scales[j]
                scaled_logs = np.clip(scaled_logs, -LOG_RET_CLIP, LOG_RET_CLIP)
                pred_prices = ref_prices[idx] * np.exp(scaled_logs)

                metrics = calculate_metrics(actual_prices, pred_prices)
                metrics['company'] = company
                metrics['split'] = split_idx
                results.append(metrics)

    if results:
        results_df = pd.DataFrame(results)
        output_path = OUTPUT_DIR / "xgboost_results.csv"
        results_df.to_csv(output_path, index=False)

        company_summary = (
            results_df.groupby('company')[['mae', 'rmse', 'mape', 'r2']]
            .mean()
            .reset_index()
            .sort_values('rmse', ascending=False)
        )
        summary_path = OUTPUT_DIR / "xgboost_company_summary.csv"
        company_summary.to_csv(summary_path, index=False)

        avg_metrics = results_df[['mae', 'rmse', 'mape', 'r2']].mean()
        print("\nAverage XGBoost Performance (on Prices):")
        print(avg_metrics)
        print("\nPer-company summary (worst RMSE first):")
        print(company_summary)
        print(f"\nDetailed results saved to {output_path}")
        print(f"Company summary saved to {summary_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_xgboost_analysis()

