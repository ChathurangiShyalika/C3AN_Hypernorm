"""
Direction pipeline for multi-horizon UP/FLAT/DOWN classification.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

try:
    from xgboost import XGBClassifier
except Exception: 
    XGBClassifier = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

LABELS = { -1: 'down', 0: 'flat', 1: 'up' }
TECH_FEATURES = [
    'rsi_14',
    'macd',
    'macd_signal',
    'macd_hist',
    'bb_upper',
    'bb_middle',
    'bb_lower',
    'atr_14',
    'rolling_vol_20',
    'roc_5',
    'lag_ret_1',
    'lag_ret_3',
    'lag_ret_5',
    'streak_5',
    'rel_vol_20',
    'log_ret_vol',
]


def _json_serializer(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _progress(iterable, **kwargs):
    if tqdm is None:
        raise ImportError("tqdm is required for progress output. Install it with 'pip install tqdm'.")
    return tqdm(iterable, **kwargs)


class LogisticRandomForestEnsemble:
    """Simple probability-blending ensemble of logistic regression and Random Forest."""

    def __init__(self, balance_classes: bool = True, logistic_weight: float = 0.5):
        self.balance_classes = balance_classes
        self.logistic_weight = float(np.clip(logistic_weight, 0.0, 1.0))
        self.logistic_model = _make_logistic_pipeline(balance_classes)
        self.rf_model = _make_random_forest(balance_classes)
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.logistic_model.fit(X, y)
        self.rf_model.fit(X, y)
        self.classes_ = self.logistic_model.classes_
        # Align class order by taking union to be safe
        rf_classes = self.rf_model.classes_
        if not np.array_equal(self.classes_, rf_classes):
            union = np.unique(np.concatenate([self.classes_, rf_classes]))
            self.classes_ = union
        return self

    def _align_proba(self, model, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError('Ensemble must be fit before calling predict_proba.')
        proba = model.predict_proba(X)
        aligned = np.zeros((len(X), len(self.classes_)))
        model_classes = model.classes_
        for idx, cls in enumerate(model_classes):
            matches = np.where(self.classes_ == cls)[0]
            if len(matches) == 0:
                continue
            target_idx = matches[0]
            aligned[:, target_idx] = proba[:, idx]
        return aligned

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_lr = self._align_proba(self.logistic_model, X)
        p_rf = self._align_proba(self.rf_model, X)
        return self.logistic_weight * p_lr + (1.0 - self.logistic_weight) * p_rf

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


class TemporalConvNetClassifier:
    """Lightweight Temporal Convolutional classifier implemented with PyTorch."""

    def __init__(
        self,
        balance_classes: bool = True,
        hidden_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 50,
        patience: int = 6,
        random_state: int = 42,
        device: str = 'auto',
    ):
        self.balance_classes = balance_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device_choice = device

        self.model: nn.Module | None = None
        self.classes_: np.ndarray | None = None
        self.class_to_idx: Dict[int, int] | None = None
        self.device = None
        self.input_dim_: int | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None

    def _check_backend(self):
        if torch is None or nn is None:
            raise ImportError(
                "PyTorch is required for model_type='tcn'. Please install torch>=1.13 to continue."
            )

    def _set_random_seeds(self):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _resolve_device(self) -> torch.device:
        choice = (self.device_choice or 'auto').lower()
        if choice == 'cpu':
            return torch.device('cpu')
        if choice == 'cuda':
            if not torch.cuda.is_available():
                warnings.warn(
                    'CUDA was requested but is not available in this environment. Falling back to CPU.',
                    RuntimeWarning,
                )
                return torch.device('cpu')
            return torch.device('cuda')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        layers: List[nn.Module] = []
        in_channels = 1
        dilation = 1
        for _ in range(self.num_layers):
            padding = (self.kernel_size - 1) * dilation // 2
            conv = nn.Conv1d(
                in_channels,
                self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=padding,
                dilation=dilation,
            )
            layers.extend([
                conv,
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            in_channels = self.hidden_channels
            dilation *= 2
        net = nn.Sequential(*layers)

        class Head(nn.Module):
            def __init__(self, backbone: nn.Module, hidden: int, n_classes: int):
                super().__init__()
                self.backbone = backbone
                self.fc = nn.Linear(hidden, n_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.backbone(x)
                pooled = features.mean(dim=-1)
                return self.fc(pooled)

        return Head(net, self.hidden_channels, n_classes)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._check_backend()
        self._set_random_seeds()
        self.device = self._resolve_device()

        X_np = np.asarray(X, dtype=np.float32)
        self.feature_mean_ = X_np.mean(axis=0)
        self.feature_std_ = X_np.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ == 0.0, 1.0, self.feature_std_)
        X_np = (X_np - self.feature_mean_) / self.feature_std_
        y_np = np.asarray(y)
        self.classes_ = np.unique(y_np)
        self.class_to_idx = {int(cls): idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx[int(val)] for val in y_np], dtype=np.int64)

        self.input_dim_ = X_np.shape[1]
        n_classes = len(self.classes_)
        self.model = self._build_model(self.input_dim_, n_classes).to(self.device)

        dataset = TensorDataset(
            torch.from_numpy(X_np),
            torch.from_numpy(y_idx),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if self.balance_classes and n_classes > 1:
            class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
            class_weights = class_counts.sum() / (n_classes * np.clip(class_counts, 1.0, None))
            weight_tensor = torch.from_numpy(class_weights).float().to(self.device)
        else:
            weight_tensor = None

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in _progress(range(self.max_epochs), desc='TCN epochs', leave=False):
            self.model.train()
            running_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device).float().unsqueeze(1)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
            avg_loss = running_loss / max(1, len(loader))
            if avg_loss + 1e-5 < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None and self.model is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.model.eval()
        return self

    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None:
            raise RuntimeError('TemporalConvNetClassifier must be fit before prediction.')
        X_np = np.asarray(X, dtype=np.float32)
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError('Normalization stats not set for TemporalConvNetClassifier.')
        X_np = (X_np - self.feature_mean_) / self.feature_std_
        dataset = TensorDataset(torch.from_numpy(X_np))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        logits_list: List[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device).float().unsqueeze(1)
                logits = self.model(xb)
                logits_list.append(logits.cpu().numpy())
        return np.concatenate(logits_list, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(X)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        if self.classes_ is None:
            raise RuntimeError('Classes not set for TemporalConvNetClassifier.')
        inv_map = np.array(self.classes_)
        return inv_map[idx]


@dataclass
class DirectionConfig:
    horizons: Tuple[int, ...] = (1,2,3,4,5)
    n_lags: int = 10
    price_col: str = 'ohlc_avg'
    company_col: str = 'company'
    date_col: str = 'date'
    flat_threshold_pct: float = 0.2
    test_size: float = 0.2
    n_splits_cv: int = 5
    balance_classes: bool = True
    use_technical_indicators: bool = True
    model_type: str = 'logistic'
    rsi_period: int = 14
    aggregate_daily: bool = True
    flat_threshold_strategy: str = 'static'  # options: static, sqrt_h, atr, atr_sqrt
    atr_threshold_multiplier: float = 1.0
    atr_up_multiplier: float = 1.0   # Asymmetric factor: k_u
    atr_down_multiplier: float = 0.8 # Asymmetric factor: k_d (k_d < k_u)
    ensemble_logistic_weight: float = 0.5
    prediction_margin_threshold: float = 0.1
    log_run_metadata: bool = True
    tcn_hidden_channels: int = 64
    tcn_num_layers: int = 3
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1
    tcn_lr: float = 1e-3
    tcn_batch_size: int = 256
    tcn_max_epochs: int = 50
    tcn_patience: int = 6
    tcn_device: str = 'auto'


def _config_to_serializable(cfg: DirectionConfig) -> Dict[str, Any]:
    data = asdict(cfg)
    data['horizons'] = list(cfg.horizons)
    return data


def _ensure_price_col(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    df = df.copy()
    if price_col not in df.columns:
        required = {'open','high','low','close'}
        if not required.issubset(df.columns):
            raise ValueError(
                f"price_col='{price_col}' missing and cannot be built; missing OHLC columns"
            )
        df[price_col] = df[['open','high','low','close']].mean(axis=1)
    return df


def _sort_df(df: pd.DataFrame, company_col: str, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.sort_values([company_col, date_col]).reset_index(drop=True)


def _make_lag_features(series: pd.Series, n_lags: int) -> pd.DataFrame:
    data = { f'lag_{i}': series.shift(i) for i in range(n_lags) }
    return pd.DataFrame(data)


def _compute_balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    return class_weight.compute_sample_weight('balanced', y)


def _aggregate_daily(df: pd.DataFrame, cfg: DirectionConfig) -> pd.DataFrame:
    df = df.sort_values([cfg.company_col, cfg.date_col]).reset_index(drop=True)
    out = (
        df.groupby([cfg.company_col, cfg.date_col], observed=True)
        .agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum'),
        )
        .reset_index()
    )
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col])
    out[cfg.company_col] = out[cfg.company_col].astype('category')
    return out


def _add_technical_indicators(df: pd.DataFrame, cfg: DirectionConfig) -> pd.DataFrame:
    out = df.copy()
    price = out[cfg.price_col].astype(float)

    # RSI (Wilder's smoothing)
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / cfg.rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / cfg.rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    # ema_fast = price.ewm(span=12, adjust=False).mean()
    # ema_slow = price.ewm(span=26, adjust=False).mean()
    # out['macd'] = ema_fast - ema_slow
    # out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    # out['macd_hist'] = out['macd'] - out['macd_signal']

    # Bollinger Bands (20-period)
    rolling_mean = price.rolling(window=20, min_periods=20).mean()
    rolling_std = price.rolling(window=20, min_periods=20).std()
    out['bb_middle'] = rolling_mean
    out['bb_upper'] = rolling_mean + 2 * rolling_std
    out['bb_lower'] = rolling_mean - 2 * rolling_std

    # ATR (Average True Range)
    high = out['high'].astype(float)
    low = out['low'].astype(float)
    close = out['close'].astype(float)
    prev_close = close.shift(1)
    tr_components = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    true_range = tr_components.max(axis=1)
    out['atr_14'] = true_range.rolling(window=14, min_periods=14).mean()

    pct = price.pct_change(fill_method=None)
    out['rolling_vol_20'] = pct.rolling(window=20, min_periods=20).std()
    # out['roc_5'] = price.pct_change(periods=5, fill_method=None)
    out['lag_ret_1'] = pct
    out['lag_ret_3'] = price.pct_change(periods=3, fill_method=None)
    out['lag_ret_5'] = price.pct_change(periods=5, fill_method=None)

    delta = price.diff().fillna(0)
    streak = np.sign(delta)
    # out['streak_5'] = streak.rolling(window=5, min_periods=5).sum()

    # Volume Features
    if 'volume' in out.columns:
        vol = out['volume'].astype(float).replace(0, np.nan)
        # Relative Volume: (Current Vol / 20-day Avg Vol) - Centered around 1.0
        out['rel_vol_20'] = vol / vol.rolling(window=20, min_periods=20).mean()
        # Log Volume Change: Stationary measure of volume surges
        out['log_ret_vol'] = np.log(vol).diff()
    
    return out


def _threshold_requires_tech(cfg: DirectionConfig) -> bool:
    strategy = (cfg.flat_threshold_strategy or 'static').lower()
    return 'atr' in strategy


def _compute_flat_threshold_series(
    working: pd.DataFrame,
    cfg: DirectionConfig,
    horizon: int,
) -> Tuple[pd.Series, pd.Series]:
    """Return (up_threshold_series, down_threshold_series)"""
    base = pd.Series(cfg.flat_threshold_pct, index=working.index, dtype=float)
    strategy = (cfg.flat_threshold_strategy or 'static').lower()

    time_scale = 1.0
    if 'sqrt' in strategy:
        time_scale = np.sqrt(max(1, horizon))
    
    base = base * time_scale
    
    base_up = base.copy()
    base_down = base.copy()

    if 'atr' in strategy:
        if 'atr_14' not in working.columns:
            raise ValueError(
                "ATR-based threshold selected but 'atr_14' not present. Enable technical indicators or choose a different strategy."
            )
        price = working[cfg.price_col].astype(float).replace(0, np.nan)
        atr_pct = (working['atr_14'].astype(float) / price).abs() * 100.0
        
        base_atr = atr_pct * time_scale
        
        # Apply asymmetric factors: k_u for Up, k_d for Down
        k_u = getattr(cfg, 'atr_up_multiplier', cfg.atr_threshold_multiplier)
        k_d = getattr(cfg, 'atr_down_multiplier', cfg.atr_threshold_multiplier)
        
        scaled_up = base_atr * k_u
        scaled_down = base_atr * k_d
        
        base_up = pd.concat([base_up, scaled_up], axis=1).max(axis=1)
        base_down = pd.concat([base_down, scaled_down], axis=1).max(axis=1)
    else:
        pass

    return (
        base_up.fillna(cfg.flat_threshold_pct), 
        base_down.fillna(cfg.flat_threshold_pct)
    )


def _label_direction(
    base: pd.Series,
    future: pd.Series,
    thresholds: Tuple[pd.Series | float, pd.Series | float],
    min_threshold: float,
) -> pd.Series:
    pct = (future / base - 1.0) * 100.0
    
    thr_up, thr_down = thresholds
    
    if isinstance(thr_up, pd.Series):
        thr_up_vals = thr_up.reindex_like(pct).ffill().bfill().fillna(min_threshold).to_numpy()
        thr_down_vals = thr_down.reindex_like(pct).ffill().bfill().fillna(min_threshold).to_numpy()
    else:
        thr_up_vals = float(thr_up)
        thr_down_vals = float(thr_down)
        
    # Up if pct > thr_up
    # Down if pct < -thr_down
    labels = np.where(pct > thr_up_vals, 1, 
                      np.where(pct < -thr_down_vals, -1, 0))
    return pd.Series(labels, index=base.index)


def _build_supervised_for_company(
    cdf: pd.DataFrame,
    cfg: DirectionConfig,
) -> Dict[int, pd.DataFrame]:
    working = cdf.copy()
    if cfg.use_technical_indicators or _threshold_requires_tech(cfg):
        working = _add_technical_indicators(working, cfg)
    s = working[cfg.price_col].astype(float)
    lags = _make_lag_features(s, cfg.n_lags)
    tech_cols = [col for col in TECH_FEATURES if col in working.columns]

    out = {}
    for h in cfg.horizons:
        target = s.shift(-h)
        thresholds_tuple = _compute_flat_threshold_series(working, cfg, h)
        y = _label_direction(base=s, future=target, thresholds=thresholds_tuple, min_threshold=cfg.flat_threshold_pct)
        
        meta = working[[cfg.date_col, cfg.company_col]].copy()
        meta['target_date'] = working[cfg.date_col].shift(-h)
        meta['horizon'] = h

        frames = [
            meta,
            lags,
            working[tech_cols] if tech_cols else None,
            y.rename('y'),
        ]
        out[h] = pd.concat([f for f in frames if f is not None], axis=1).dropna()

    return out


def _chronological_split(n: int, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    n_test = max(1, int(round(n * test_size)))
    n_train = max(1, n - n_test)
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:]


def _make_random_forest(balance: bool) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced' if balance else None,
        n_jobs=-1,
    )


def _make_xgboost(balance: bool) -> Any:
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Please install it to use model_type='xgboost'.")
    params = dict(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        learning_rate=0.05,
        n_estimators=600,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
    )
    return XGBClassifier(**params)


def _make_tcn_classifier(cfg: DirectionConfig) -> TemporalConvNetClassifier:
    if torch is None:
        raise ImportError("PyTorch is required for model_type='tcn'. Please install torch before using this option.")
    return TemporalConvNetClassifier(
        balance_classes=cfg.balance_classes,
        hidden_channels=cfg.tcn_hidden_channels,
        num_layers=cfg.tcn_num_layers,
        kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.tcn_dropout,
        lr=cfg.tcn_lr,
        batch_size=cfg.tcn_batch_size,
        max_epochs=cfg.tcn_max_epochs,
        patience=cfg.tcn_patience,
        device=cfg.tcn_device,
    )


def _make_logistic_pipeline(balance: bool) -> Pipeline:
    if balance:
        lr = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
    else:
        lr = LogisticRegression(max_iter=2000, solver='lbfgs')
    return Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', lr)
    ])


def _clf_pipeline(cfg: DirectionConfig):
    model_type = (cfg.model_type or 'logistic').lower()
    if model_type == 'random_forest':
        return _make_random_forest(cfg.balance_classes)
    if model_type == 'xgboost':
        return _make_xgboost(cfg.balance_classes)
    if model_type == 'ensemble':
        return LogisticRandomForestEnsemble(
            balance_classes=cfg.balance_classes,
            logistic_weight=cfg.ensemble_logistic_weight,
        )
    if model_type == 'tcn':
        return _make_tcn_classifier(cfg)
    return _make_logistic_pipeline(cfg.balance_classes)


def _evaluate_company_horizon(Xy: pd.DataFrame, cfg: DirectionConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        c for c in Xy.columns
        if c.startswith('lag_') or c in TECH_FEATURES
    ]
    X = Xy[feature_cols].values
    y = Xy['y'].values

    is_xgb = cfg.model_type == 'xgboost'
    if is_xgb:
        label_to_internal = {-1: 0, 0: 1, 1: 2}
        internal_to_label = {v: k for k, v in label_to_internal.items()}
        y_internal = np.array([label_to_internal[int(val)] for val in y], dtype=int)
    else:
        label_to_internal = None
        internal_to_label = None
        y_internal = y

    train_idx, test_idx = _chronological_split(len(Xy), cfg.test_size)

    model = _clf_pipeline(cfg)
    train_sample_weight = None
    if is_xgb and cfg.balance_classes:
        train_sample_weight = _compute_balanced_sample_weights(y_internal[train_idx])
    
    if train_sample_weight is not None:
        model.fit(X[train_idx], y_internal[train_idx], sample_weight=train_sample_weight)
    else:
        model.fit(X[train_idx], y_internal[train_idx])

    try:
        importance_df = _extract_feature_importance(model, feature_cols)
        importance_df['company'] = Xy[cfg.company_col].iloc[0]
        importance_df['horizon'] = Xy['horizon'].iloc[0]
    except Exception as e:
        print(f"Warning: Could not extract importance: {e}")
        importance_df = pd.DataFrame()


    tscv = TimeSeriesSplit(n_splits=min(cfg.n_splits_cv, max(2, len(Xy)//20)))
    cv_accs, cv_f1s = [], []
    for tr, va in tscv.split(X):
        if len(np.unique(y[tr])) < 2: 
            continue
        m = _clf_pipeline(cfg)
        sw = None
        if is_xgb and cfg.balance_classes:
            sw = _compute_balanced_sample_weights(y_internal[tr])
        if sw is not None:
            m.fit(X[tr], y_internal[tr], sample_weight=sw)
        else:
            m.fit(X[tr], y_internal[tr])
        p_internal = m.predict(X[va])
        if is_xgb:
            p = np.array([internal_to_label[int(val)] for val in p_internal])
        else:
            p = p_internal
        cv_accs.append(accuracy_score(y[va], p))
        cv_f1s.append(f1_score(y[va], p, average='macro'))

    # Test metrics
    y_pred_internal = model.predict(X[test_idx])
    if is_xgb:
        y_pred = np.array([internal_to_label[int(val)] for val in y_pred_internal])
    else:
        y_pred = y_pred_internal

    classes_for_proba = None
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba_internal = model.predict_proba(X[test_idx])
        y_proba = y_proba_internal
        if is_xgb:
            classes_raw = getattr(model, 'classes_', np.array([0, 1, 2]))
            classes_for_proba = np.array([internal_to_label[int(cls)] for cls in classes_raw])
        else:
            classes_for_proba = getattr(model, 'classes_', np.array([-1, 0, 1]))
    else:
        classes_for_proba = np.array([-1, 0, 1])

    n_test = len(test_idx)
    final_pred = y_pred.copy()
    operable_flags = np.zeros(n_test, dtype=bool)
    margin_values = np.full(n_test, np.nan)
    dir_prob_values = np.full(n_test, np.nan)
    p_down_arr = np.full(n_test, np.nan)
    p_flat_arr = np.full(n_test, np.nan)
    p_up_arr = np.full(n_test, np.nan)

    if y_proba is not None and classes_for_proba is not None and n_test > 0:
        class_to_idx = {int(cls): idx for idx, cls in enumerate(classes_for_proba)}

        def _get_probs(label: int) -> np.ndarray:
            idx = class_to_idx.get(label)
            if idx is None:
                return np.zeros(n_test, dtype=float)
            return y_proba[:, idx]

        p_down_arr = _get_probs(-1)
        p_flat_arr = _get_probs(0)
        p_up_arr = _get_probs(1)

        dir_is_up = p_up_arr >= p_down_arr
        dir_labels = np.where(dir_is_up, 1, -1)
        dir_prob_values = np.where(dir_is_up, p_up_arr, p_down_arr)
        margin_values = dir_prob_values - p_flat_arr
        operable_flags = margin_values >= cfg.prediction_margin_threshold
        final_pred = np.where(operable_flags, dir_labels, 0)

    acc = accuracy_score(y[test_idx], final_pred)
    f1m = f1_score(y[test_idx], final_pred, average='macro')

    cm = confusion_matrix(y[test_idx], final_pred, labels=[-1,0,1], normalize='true')
    cm_df = pd.DataFrame(cm, index=[LABELS[-1], LABELS[0], LABELS[1]], columns=[LABELS[-1], LABELS[0], LABELS[1]])

    pred_rows = []
    for i, row_idx in enumerate(test_idx):
        row = Xy.iloc[row_idx]
        if y_proba is not None and classes_for_proba is not None:
            proba_map = { int(cls): float(prob) for cls, prob in zip(classes_for_proba, y_proba[i]) }
        else:
            proba_map = {}
        true_label_str = LABELS[int(row['y'])]
        pred_label_str = LABELS[int(final_pred[i])] if i < len(final_pred) else LABELS[int(y_pred[i])]
        operable_flag = bool(operable_flags[i]) if i < len(operable_flags) else False
        margin_val = float(margin_values[i]) if i < len(margin_values) else np.nan
        dir_prob_val = float(dir_prob_values[i]) if i < len(dir_prob_values) else np.nan
        pred_rows.append({
            'company': row['company'],
            'base_date': row['date'],
            'target_date': row['target_date'],
            'horizon': row['horizon'],
            'true_label': true_label_str,
            'pred_label': pred_label_str,
            'proba_down': float(p_down_arr[i]) if i < len(p_down_arr) else proba_map.get(-1, np.nan),
            'proba_flat': float(p_flat_arr[i]) if i < len(p_flat_arr) else proba_map.get(0, np.nan),
            'proba_up': float(p_up_arr[i]) if i < len(p_up_arr) else proba_map.get(1, np.nan),
            'proba_dir': dir_prob_val,
            'proba_margin': margin_val,
            'operable': operable_flag,
            'is_correct': true_label_str == pred_label_str,
        })
    preds_df = pd.DataFrame(pred_rows)

    metrics_row = {
        'company': Xy['company'].iloc[0],
        'n_train': int((train_idx[-1] + 1) if len(train_idx) else 0),
        'n_test': int(len(test_idx)),
        'acc_test': float(acc),
        'f1_macro_test': float(f1m),
        'acc_cv_mean': float(np.mean(cv_accs)) if cv_accs else np.nan,
        'f1_macro_cv_mean': float(np.mean(cv_f1s)) if cv_f1s else np.nan,
        'horizon': int(Xy['horizon'].iloc[0]),
        'flat_threshold_pct': float(cfg.flat_threshold_pct),
    }
    metrics_df = pd.DataFrame([metrics_row])

    return metrics_df, cm_df, preds_df, importance_df


def _extract_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    importances = np.zeros(len(feature_names))
    counts = 0

    def process_estimator(est):
        nonlocal counts
        if hasattr(est, 'feature_importances_'):
            nonlocal importances
            importances += est.feature_importances_
            counts += 1
        elif hasattr(est, 'coef_'):
            importances += np.abs(est.coef_).mean(axis=0)
            counts += 1
        elif hasattr(est, 'steps'):
            process_estimator(est.steps[-1][1])
    
    if isinstance(model, LogisticRandomForestEnsemble):
        process_estimator(model.rf_model)
        process_estimator(model.logistic_model)
    else:
        process_estimator(model)

    if counts > 0:
        importances /= counts
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)


def run_direction_pipeline(
    df: pd.DataFrame,
    output_dir: Path | str,
    horizons: Iterable[int] = (1,2,3,4,5),
    n_lags: int = 10,
    price_col: str = 'ohlc_avg',
    company_col: str = 'company',
    date_col: str = 'date',
    flat_threshold_pct: float = 0.2,
    test_size: float = 0.2,
    n_splits_cv: int = 5,
    balance_classes: bool = True,
    use_technical_indicators: bool = True,
    model_type: str = 'logistic',
    aggregate_daily: bool = True,
    flat_threshold_strategy: str = 'static',
    atr_threshold_multiplier: float = 1.0,
    atr_up_multiplier: float = 1.0,
    atr_down_multiplier: float = 0.8,
    ensemble_logistic_weight: float = 0.5,
    prediction_margin_threshold: float = 0.1,
    log_run_metadata: bool = True,
    tcn_hidden_channels: int = 64,
    tcn_num_layers: int = 3,
    tcn_kernel_size: int = 3,
    tcn_dropout: float = 0.1,
    tcn_lr: float = 1e-3,
    tcn_batch_size: int = 256,
    tcn_max_epochs: int = 50,
    tcn_patience: int = 6,
    tcn_device: str = 'auto',
) -> None:
    cfg = DirectionConfig(
        horizons=tuple(int(h) for h in horizons),
        n_lags=int(n_lags),
        price_col=price_col,
        company_col=company_col,
        date_col=date_col,
        flat_threshold_pct=float(flat_threshold_pct),
        test_size=float(test_size),
        n_splits_cv=int(n_splits_cv),
        balance_classes=bool(balance_classes),
        use_technical_indicators=bool(use_technical_indicators),
        model_type=model_type,
        aggregate_daily=bool(aggregate_daily),
        flat_threshold_strategy=flat_threshold_strategy,
        atr_threshold_multiplier=float(atr_threshold_multiplier),
        atr_up_multiplier=float(atr_up_multiplier),
        atr_down_multiplier=float(atr_down_multiplier),
        ensemble_logistic_weight=float(ensemble_logistic_weight),
        prediction_margin_threshold=float(prediction_margin_threshold),
        log_run_metadata=bool(log_run_metadata),
        tcn_hidden_channels=int(tcn_hidden_channels),
        tcn_num_layers=int(tcn_num_layers),
        tcn_kernel_size=int(tcn_kernel_size),
        tcn_dropout=float(tcn_dropout),
        tcn_lr=float(tcn_lr),
        tcn_batch_size=int(tcn_batch_size),
        tcn_max_epochs=int(tcn_max_epochs),
        tcn_patience=int(tcn_patience),
        tcn_device=tcn_device,
    )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.aggregate_daily:
        df = _aggregate_daily(df, cfg)

    df = _ensure_price_col(df, cfg.price_col)
    df = _sort_df(df, cfg.company_col, cfg.date_col)

    companies = df[cfg.company_col].dropna().unique().tolist()

    all_metrics: List[pd.DataFrame] = []
    all_importances = []
    overall_cm_by_h: Dict[int, np.ndarray] = {}
    per_company_cm_rows: List[Dict] = []
    operability_stats: List[Dict[str, Any]] = []

    for h in _progress(cfg.horizons, desc='Horizons'):
        preds_all = []
        cms_sum = np.zeros((3,3), dtype=float)
        total_weight = 0.0

        for comp in _progress(companies, desc=f'H{h} companies', total=len(companies)):
            cdf = df[df[cfg.company_col] == comp].reset_index(drop=True)
            if len(cdf) < (cfg.n_lags + h + 20):
                # not enough history
                continue
            per_h = _build_supervised_for_company(cdf, cfg)[h]
            if per_h.empty:
                continue

            metrics_df, cm_df, preds_df, import_df = _evaluate_company_horizon(per_h, cfg)
            all_metrics.append(metrics_df)
            preds_all.append(preds_df)
            all_importances.append(import_df)

            n_test = metrics_df['n_test'].iloc[0]
            cms_sum += cm_df.values * n_test
            total_weight += n_test

            row = cm_df.copy()
            row.insert(0, 'company', comp)
            row.insert(1, 'horizon', h)
            per_company_cm_rows.append(row)

        if preds_all:
            preds_concat = pd.concat(preds_all, ignore_index=True)
            preds_concat.to_csv(outdir / f'direction_preds_h{h}.csv', index=False)

            operable_mask = preds_concat['operable']
            operable_n = int(operable_mask.sum())
            total_preds = int(len(preds_concat))
            operable_accuracy = float(preds_concat.loc[operable_mask, 'is_correct'].mean()) if operable_n > 0 else np.nan
            operability_stats.append({
                'horizon': h,
                'total_predictions': total_preds,
                'operable_predictions': operable_n,
                'operable_share': float(operable_n / total_preds) if total_preds > 0 else np.nan,
                'operable_accuracy': operable_accuracy,
            })


        if total_weight > 0:
            overall = cms_sum / total_weight
            overall_cm_by_h[h] = overall
            overall_df = pd.DataFrame(overall, index=[LABELS[-1], LABELS[0], LABELS[1]], columns=[LABELS[-1], LABELS[0], LABELS[1]])
            overall_df.to_csv(outdir / f'direction_confusion_overall_h{h}.csv')

    metrics_summary_records: List[Dict[str, Any]] = []
    if all_metrics:
        metrics_concat = pd.concat(all_metrics, ignore_index=True)
        metrics_concat.to_csv(outdir / 'direction_classification_metrics.csv', index=False)
        metrics_summary_records = (
            metrics_concat
            .groupby('horizon')[['acc_test', 'f1_macro_test', 'acc_cv_mean', 'f1_macro_cv_mean']]
            .mean()
            .reset_index()
            .to_dict('records')
        )
    
    if all_importances:
        imp_df = pd.concat(all_importances, ignore_index=True)
        summary = imp_df.groupby(['horizon', 'feature'])['importance'].mean().reset_index()
        summary = summary.sort_values(['horizon', 'importance'], ascending=[True, False])
        summary.to_csv(outdir / 'feature_importance_summary.csv', index=False)
    
    if per_company_cm_rows:
        pd.concat(per_company_cm_rows, ignore_index=True).to_csv(outdir / 'direction_confusion_per_company.csv', index=True)

    with open(outdir / 'direction_readme.txt', 'w') as fh:
        fh.write(
            'Direction classification pipeline\n'
            f'Model type: {cfg.model_type}\n'
            f'Use technical indicators: {cfg.use_technical_indicators}\n'
            f'Aggregate daily: {cfg.aggregate_daily}\n'
            f'Horizons: {list(cfg.horizons)}\n'
            f'Lags: {cfg.n_lags}\n'
            f'Price column: {cfg.price_col}\n'
            f'Flat threshold (%): {cfg.flat_threshold_pct}\n'
            f'Flat threshold strategy: {cfg.flat_threshold_strategy}\n'
            f'ATR threshold multiplier: {cfg.atr_threshold_multiplier}\n'
            f'Ensemble logistic weight: {cfg.ensemble_logistic_weight}\n'
            f'Prediction margin threshold: {cfg.prediction_margin_threshold}\n'
            f'Log run metadata: {cfg.log_run_metadata}\n'
            f'Test size (proportion): {cfg.test_size}\n'
            'Labels: -1=down, 0=flat, 1=up. Confusion matrices are row-normalized by true class.\n'
        )

    if cfg.log_run_metadata:
        summary_payload = {
            'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
            'output_dir': str(outdir.resolve()),
            'config': _config_to_serializable(cfg),
            'metrics_summary': metrics_summary_records,
            'operability_summary': operability_stats,
            'files_written': sorted([p.name for p in outdir.glob('*') if p.is_file()]),
        }
        with open(outdir / 'run_summary.json', 'w') as fh:
            json.dump(summary_payload, fh, indent=2, default=_json_serializer)
