import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Union, Generator, Tuple
from pathlib import Path

def calculate_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate common regression metrics: MAE, RMSE, MAPE, R2.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if np.isclose(y_true.std(), 0.0, atol=1e-12):
        r2 = float('nan')
    else:
        r2 = r2_score(y_true, y_pred)
    
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except ValueError:
        mape = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }

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

    df = df.dropna(subset=["open", "high", "low", "close"])
    
    df = df.drop_duplicates(subset=['company', 'date'], keep='first')
    
    df = df.sort_values(["company", "date"])
    return df

class WalkForwardValidator:
    """
    Helper for Time Series Cross Validation using Expanding Window.
    """
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def split(self, X) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields train_index, test_index using TimeSeriesSplit (expanding window).
        """
        return self.tscv.split(X)

def make_lagged_dataset(
    group: pd.DataFrame, n_lags: int = 10, horizon: int = 5
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
        g[f"lag_{lag}"] = g["close"].shift(lag - 1)  

    for h in range(1, horizon + 1):
        g[f"target_{h}"] = g["close"].shift(-h)  
        g[f"target_date_{h}"] = g["date"].shift(-h)
        g[f"target_close_h{h}"] = g["close"].shift(-h) 

    cols_to_check = [f"lag_{i}" for i in range(1, n_lags + 1)] + [
        f"target_{h}" for h in range(1, horizon + 1)
    ] + [f"target_close_h{h}" for h in range(1, horizon + 1)]
    g = g.dropna(subset=cols_to_check).reset_index(drop=True)
    return g

def create_price_feature_frame(
    group: pd.DataFrame,
    horizon: int = 5,
    n_price_lags: int = 15,
    n_return_lags: int = 10,
    momentum_windows = (3, 5, 10, 20)
) -> Tuple[pd.DataFrame, list, list]:
    """Build an enriched feature frame for tree models forecasting prices.

    Features include:
      - Close price lags (lag_close_i)
      - Return lags (lag_ret_i)
      - Rolling statistics over prices/returns
      - Momentum ratios and volume deltas
      - High-low range based volatility proxies

    Targets are future closes for the next `horizon` days.
    Returns (frame, feature_cols, target_cols).
    """
    g = group.copy().sort_values("date").reset_index(drop=True)

    g["return_1"] = g["close"].pct_change()
    g["log_return_1"] = np.log(g["close"]).diff()

    feature_cols = []

    for lag in range(1, n_price_lags + 1):
        col = f"lag_close_{lag}"
        g[col] = g["close"].shift(lag)
        feature_cols.append(col)

    for lag in range(1, n_return_lags + 1):
        col = f"lag_ret_{lag}"
        g[col] = g["return_1"].shift(lag)
        feature_cols.append(col)

    # Rolling statistics on price and returns (shifted to avoid leakage)
    rolling_windows = {3, 5, 10, 20}
    for window in rolling_windows:
        mean_col = f"roll_close_mean_{window}"
        std_col = f"roll_close_std_{window}"
        rmean_col = f"roll_ret_mean_{window}"
        rstd_col = f"roll_ret_std_{window}"
        g[mean_col] = g["close"].shift(1).rolling(window).mean()
        g[std_col] = g["close"].shift(1).rolling(window).std()
        g[rmean_col] = g["return_1"].shift(1).rolling(window).mean()
        g[rstd_col] = g["return_1"].shift(1).rolling(window).std()
        feature_cols.extend([mean_col, std_col, rmean_col, rstd_col])

    # Momentum style ratios
    for window in momentum_windows:
        col = f"momentum_{window}"
        g[col] = g["close"].divide(g["close"].shift(window)) - 1
        feature_cols.append(col)

    # Volume deltas
    for window in (3, 5, 10):
        col = f"vol_delta_{window}"
        g[col] = g["volume"].divide(g["volume"].shift(window))
        feature_cols.append(col)

    # High-low range based volatility
    g["hl_range_pct"] = (g["high"] - g["low"]) / g["close"]
    feature_cols.append("hl_range_pct")
    for window in (3, 5, 10):
        col = f"hl_range_pct_mean_{window}"
        g[col] = g["hl_range_pct"].shift(1).rolling(window).mean()
        feature_cols.append(col)

    target_cols = []
    for h in range(1, horizon + 1):
        col = f"target_close_h{h}"
        g[col] = g["close"].shift(-h)
        target_cols.append(col)

    g = g.replace([np.inf, -np.inf], np.nan)
    required = feature_cols + target_cols
    g = g.dropna(subset=required).reset_index(drop=True)
    return g, feature_cols, target_cols

def create_advanced_features(
    df: pd.DataFrame, 
    n_lags: int = 30, 
    horizon: int = 5
) -> pd.DataFrame:
    """
    Create advanced features for return prediction.
    Features:
    - Lagged returns (1 to n_lags)
    - Rolling stats (mean, std, ATR-like)
    - Volume deltas
    - Targets: 1-day returns for next `horizon` days
    """
    df = df.copy()
    df = df.sort_values(["company", "date"])
    
    # 1. Calculate 1-day Returns
    # ret_t = (Price_t - Price_{t-1}) / Price_{t-1}
    df['return'] = df.groupby('company')['close'].pct_change()
    
    # 2. Lagged Returns
    # lag_ret_1 = return_t
    # lag_ret_2 = return_{t-1}
    for i in range(1, n_lags + 1):
        df[f'lag_ret_{i}'] = df.groupby('company')['return'].shift(i-1) # shift(0) is current return
        
    # 3. Rolling Stats (on returns)
    # Computed on 'return' (which is available at time t)
    for window in [5, 10, 20]:
        df[f'roll_mean_ret_{window}'] = df.groupby('company')['return'].transform(lambda x: x.rolling(window).mean())
        df[f'roll_std_ret_{window}'] = df.groupby('company')['return'].transform(lambda x: x.rolling(window).std())
        
    # 4. ATR-like (High-Low)/Close
    # Available at time t
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    for window in [5, 10, 20]:
        df[f'roll_mean_hl_{window}'] = df.groupby('company')['hl_range'].transform(lambda x: x.rolling(window).mean())

    # 5. Volume Deltas
    # vol_t / vol_{t-5}
    df['vol_delta_5'] = df['volume'] / df.groupby('company')['volume'].shift(5)
    
    # 6. Targets
    # target_ret_h = return at t+h
    # shift(-h) of 'return' column gives return at t+h?
    # 'return' at t is (P_t - P_{t-1})/P_{t-1}.
    # We want return at t+1: (P_{t+1} - P_t)/P_t.
    # This is 'return' shifted by -1.
    for h in range(1, horizon + 1):
        df[f'target_ret_{h}'] = df.groupby('company')['return'].shift(-h)
        # Also keep price for reconstruction/validation
        df[f'target_close_{h}'] = df.groupby('company')['close'].shift(-h)
        
    df['ref_price'] = df['close']
    
    drop_cols = [f'lag_ret_{n_lags}', f'target_ret_{horizon}', f'roll_mean_ret_20']
    df = df.dropna(subset=drop_cols)
    
    return df


def create_features_and_targets(
    df: pd.DataFrame, 
    price_col: str = 'close', 
    n_lags: int = 10, 
    horizon: int = 5,
    return_type: str = 'log'
) -> pd.DataFrame:
    """
    Create features and targets based on Returns rather than raw prices.
    Features: Lags of returns, Rolling Mean/Std of returns.
    Targets: Future returns for next `horizon` steps.
    """
    data = df.copy()
    
    # 1. Calculate Returns
    if return_type == 'log':
        data['ret'] = np.log(data[price_col] / data[price_col].shift(1))
    else:
        data['ret'] = data[price_col].pct_change()
        
    # 2. Create Lag Features (on returns)
    feature_cols = []
    for i in range(1, n_lags + 1):
        col_name = f'lag_ret_{i}'
        data[col_name] = data['ret'].shift(i)
        feature_cols.append(col_name)
        
    # 3. Create Rolling Features (on returns) - Volatility & Trend
    # Rolling mean of returns (short-term trend)
    data['roll_mean_ret_5'] = data['ret'].shift(1).rolling(window=5).mean()
    data['roll_std_ret_5'] = data['ret'].shift(1).rolling(window=5).std()
    feature_cols.extend(['roll_mean_ret_5', 'roll_std_ret_5'])
    
    # 4. Create Targets (Future Returns)
    target_cols = []
    for h in range(1, horizon + 1):
        col_name = f'target_ret_{h}'
        data[col_name] = data['ret'].shift(-h)
        target_cols.append(col_name)
        # Also add explicit price targets for validation
        data[f'target_close_h{h}'] = data[price_col].shift(-h)
        
    # Keep reference price for reconstruction (Price at time t, right before prediction)
    data['ref_price'] = data[price_col]
    
    all_target_cols = target_cols + [f'target_close_h{h}' for h in range(1, horizon + 1)]
    data = data.dropna(subset=all_target_cols + feature_cols).reset_index(drop=True)
    
    return data, feature_cols, target_cols

def reconstruct_price_forecasts(ref_price: float, predicted_returns: np.ndarray, return_type: str = 'pct') -> np.ndarray:
    """
    Reconstruct price path from predicted returns.
    If return_type is 'pct', P_{t+h} = P_{t+h-1} * (1 + r_{t+h})
    If return_type is 'log', P_{t+h} = P_{t+h-1} * exp(r_{t+h})
    
    However, our target definition was:
    target_ret_h = return at t+h (1-day return)
    So we compound.
    """
    prices = []
    current_price = ref_price
    for ret in predicted_returns:
        if return_type == 'log':
            current_price = current_price * np.exp(ret)
        else:
            current_price = current_price * (1 + ret)
        prices.append(current_price)
    return np.array(prices)

def batch_reconstruct(last_prices, preds_returns, return_type='simple'):
    # last_prices: (n,) preds_returns: (n,H)
    import numpy as np
    n, H = preds_returns.shape
    res = np.zeros_like(preds_returns)
    for i in range(n):
        res[i] = reconstruct_price_forecasts(last_prices[i], preds_returns[i], return_type=return_type)
    return res

def debug_pred_true(y_true, y_pred, name=""):
    """
    Print debug info if std of y_true is 0 or very small.
    """
    if np.std(y_true) < 1e-6:
        print(f"DEBUG {name}: y_true std is almost zero: {np.std(y_true)}")
        print(f"DEBUG {name}: y_true sample: {y_true}")
        print(f"DEBUG {name}: y_pred sample: {y_pred}")
