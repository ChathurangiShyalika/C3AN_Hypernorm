"""
Direction pipeline for multi-horizon UP/FLAT/DOWN classification.

Drop-in module to add BEFORE your numeric forecasting step.

Key entrypoint: run_direction_pipeline(
    df, output_dir, horizons=(1,2,3,4,5), n_lags=10,
    price_col='ohlc_avg', company_col='company', date_col='date',
    flat_threshold_pct=0.2, test_size=0.2, n_splits_cv=5,
    balance_classes=True,
)

Inputs
------
- df : pandas.DataFrame with columns:
    ['date','company','open','high','low','close','volume']
  If `price_col` (default 'ohlc_avg') is missing, it will be created as mean(OHLC).

Outputs (saved under output_dir)
--------------------------------
- direction_classification_metrics.csv
    Per company × horizon test metrics: accuracy, macro_f1, support counts.
- direction_confusion_overall_h{H}.csv
    Normalized confusion matrix aggregated across companies for each horizon.
- direction_confusion_per_company_h{H}.csv
    One row per company with normalized confusion rates per true class.
- direction_preds_h{H}.csv
    Per-row predictions (company, base_date, target_date, true_label, pred_label, proba_*).
- direction_readme.txt
    Short explanation and class mapping.

Notes
-----
- Labels: compare price at t+H to price at t. If |Δ%| <= flat_threshold_pct → FLAT.
- Features: lagged values of `price_col` at t, t-1, ..., t-(n_lags-1).
- Splitting: chronological single split by proportion + CV (reported as mean accuracy/F1).
- Model: StandardScaler + LogisticRegression(multi_class='multinomial').
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

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
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

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
    ensemble_logistic_weight: float = 0.5


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
    ema_fast = price.ewm(span=12, adjust=False).mean()
    ema_slow = price.ewm(span=26, adjust=False).mean()
    out['macd'] = ema_fast - ema_slow
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    out['macd_hist'] = out['macd'] - out['macd_signal']

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

    # Rolling volatility & rate of change
    pct = price.pct_change(fill_method=None)
    out['rolling_vol_20'] = pct.rolling(window=20, min_periods=20).std()
    out['roc_5'] = price.pct_change(periods=5, fill_method=None)
    out['lag_ret_1'] = pct
    out['lag_ret_3'] = price.pct_change(periods=3, fill_method=None)
    out['lag_ret_5'] = price.pct_change(periods=5, fill_method=None)

    delta = price.diff().fillna(0)
    streak = np.sign(delta)
    # ...existing code...
    out['streak_5'] = streak.rolling(window=5, min_periods=5).sum()

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
) -> pd.Series:
    base = pd.Series(cfg.flat_threshold_pct, index=working.index, dtype=float)
    strategy = (cfg.flat_threshold_strategy or 'static').lower()

    if 'sqrt' in strategy:
        base = base * np.sqrt(max(1, horizon))

    if 'atr' in strategy:
        if 'atr_14' not in working.columns:
            raise ValueError(
                "ATR-based threshold selected but 'atr_14' not present. Enable technical indicators or choose a different strategy."
            )
        price = working[cfg.price_col].astype(float).replace(0, np.nan)
        atr_pct = (working['atr_14'].astype(float) / price).abs() * 100.0
        scaled = atr_pct * float(cfg.atr_threshold_multiplier)
        base = pd.concat([base, scaled], axis=1).max(axis=1)

    return base.fillna(cfg.flat_threshold_pct)


def _label_direction(
    base: pd.Series,
    future: pd.Series,
    flat_threshold_pct: pd.Series | float,
    min_threshold: float,
) -> pd.Series:
    # percent change from base to future
    pct = (future / base - 1.0) * 100.0
    if isinstance(flat_threshold_pct, pd.Series):
        thr = flat_threshold_pct.reindex_like(pct).fillna(method='ffill').fillna(method='bfill').fillna(min_threshold)
        thr_vals = thr.to_numpy()
    else:
        thr_vals = float(flat_threshold_pct)
    labels = np.where(pct > thr_vals, 1, np.where(pct < -thr_vals, -1, 0))
    return pd.Series(labels, index=base.index)


def _build_supervised_for_company(
    cdf: pd.DataFrame,
    cfg: DirectionConfig,
) -> Dict[int, pd.DataFrame]:
    # Build a dict of horizon -> supervised dataframe
    working = cdf.copy()
    if cfg.use_technical_indicators or _threshold_requires_tech(cfg):
        working = _add_technical_indicators(working, cfg)
    s = working[cfg.price_col].astype(float)
    lags = _make_lag_features(s, cfg.n_lags)
    tech_cols = [col for col in TECH_FEATURES if col in working.columns]

    out = {}
    for h in cfg.horizons:
        target = s.shift(-h)
        thresholds = _compute_flat_threshold_series(working, cfg, h)
        y = _label_direction(base=s, future=target, flat_threshold_pct=thresholds, min_threshold=cfg.flat_threshold_pct)
        frames = [
            working[[cfg.date_col, cfg.company_col]],
            lags,
        ]
        if tech_cols:
            frames.append(working[tech_cols])
        frames.extend([
            y.rename('y'),
            s.rename('base_price'),
            target.rename('future_price'),
        ])
        Xy = pd.concat(frames, axis=1)
        Xy['horizon'] = h
        Xy['target_date'] = working[cfg.date_col].shift(-h)
        Xy = Xy.dropna().reset_index(drop=True)
        out[h] = Xy
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


def _make_logistic_pipeline(balance: bool) -> Pipeline:
    if balance:
        lr = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
    else:
        lr = LogisticRegression(max_iter=2000, solver='lbfgs')
    return Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', lr)
    ])


def _clf_pipeline(balance: bool, model_type: str, ensemble_weight: float = 0.5):
    model_type = (model_type or 'logistic').lower()
    if model_type == 'random_forest':
        return _make_random_forest(balance)
    if model_type == 'xgboost':
        return _make_xgboost(balance)
    if model_type == 'ensemble':
        return LogisticRandomForestEnsemble(balance_classes=balance, logistic_weight=ensemble_weight)
    # default logistic regression
    return _make_logistic_pipeline(balance)


def _evaluate_company_horizon(Xy: pd.DataFrame, cfg: DirectionConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Returns: metrics_row_df, confusion_overall_df, preds_df
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

    model = _clf_pipeline(cfg.balance_classes, cfg.model_type, cfg.ensemble_logistic_weight)
    train_sample_weight = None
    if is_xgb and cfg.balance_classes:
        train_sample_weight = _compute_balanced_sample_weights(y_internal[train_idx])
    if train_sample_weight is not None:
        model.fit(X[train_idx], y_internal[train_idx], sample_weight=train_sample_weight)
    else:
        model.fit(X[train_idx], y_internal[train_idx])

    # CV with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=min(cfg.n_splits_cv, max(2, len(Xy)//20)))
    cv_accs, cv_f1s = [], []
    for tr, va in tscv.split(X):
        if len(np.unique(y[tr])) < 2:  # avoid degenerate fold
            continue
        m = _clf_pipeline(cfg.balance_classes, cfg.model_type, cfg.ensemble_logistic_weight)
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

    acc = accuracy_score(y[test_idx], y_pred)
    f1m = f1_score(y[test_idx], y_pred, average='macro')

    # Confusion matrix (normalized by true counts)
    cm = confusion_matrix(y[test_idx], y_pred, labels=[-1,0,1], normalize='true')
    cm_df = pd.DataFrame(cm, index=[LABELS[-1], LABELS[0], LABELS[1]], columns=[LABELS[-1], LABELS[0], LABELS[1]])

    # Predictions dataframe
    pred_rows = []
    for i, row_idx in enumerate(test_idx):
        row = Xy.iloc[row_idx]
        if y_proba is not None and classes_for_proba is not None:
            proba = y_proba[i]
            proba_map = { int(cls): float(prob) for cls, prob in zip(classes_for_proba, proba) }
        else:
            proba_map = {}
        pred_rows.append({
            'company': row['company'],
            'base_date': row['date'],
            'target_date': row['target_date'],
            'horizon': row['horizon'],
            'true_label': LABELS[int(row['y'])],
            'pred_label': LABELS[int(y_pred[i])],
            'proba_down': proba_map.get(-1, np.nan),
            'proba_flat': proba_map.get(0, np.nan),
            'proba_up': proba_map.get(1, np.nan),
        })
    preds_df = pd.DataFrame(pred_rows)

    # Metrics row
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

    return metrics_df, cm_df, preds_df


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
    ensemble_logistic_weight: float = 0.5,
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
        ensemble_logistic_weight=float(ensemble_logistic_weight),
    )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare dataframe
    if cfg.aggregate_daily:
        df = _aggregate_daily(df, cfg)

    df = _ensure_price_col(df, cfg.price_col)
    df = _sort_df(df, cfg.company_col, cfg.date_col)

    companies = df[cfg.company_col].dropna().unique().tolist()

    all_metrics: List[pd.DataFrame] = []
    overall_cm_by_h: Dict[int, np.ndarray] = {}
    per_company_cm_rows: List[Dict] = []

    for h in cfg.horizons:
        preds_all = []
        cms_sum = np.zeros((3,3), dtype=float)
        total_weight = 0.0

        for comp in companies:
            cdf = df[df[cfg.company_col] == comp].reset_index(drop=True)
            if len(cdf) < (cfg.n_lags + h + 20):
                # not enough history
                continue
            per_h = _build_supervised_for_company(cdf, cfg)[h]
            if per_h.empty:
                continue

            metrics_df, cm_df, preds_df = _evaluate_company_horizon(per_h, cfg)
            all_metrics.append(metrics_df)
            preds_all.append(preds_df)

            # accumulate confusion matrices weighted by number of test observations
            n_test = metrics_df['n_test'].iloc[0]
            cms_sum += cm_df.values * n_test
            total_weight += n_test

            # store per-company normalized rows
            row = cm_df.copy()
            row.insert(0, 'company', comp)
            row.insert(1, 'horizon', h)
            per_company_cm_rows.append(row)

        # write per-horizon predictions
        if preds_all:
            pd.concat(preds_all, ignore_index=True).to_csv(outdir / f'direction_preds_h{h}.csv', index=False)

        # write overall normalized confusion for this horizon
        if total_weight > 0:
            overall = cms_sum / total_weight
            overall_cm_by_h[h] = overall
            overall_df = pd.DataFrame(overall, index=[LABELS[-1], LABELS[0], LABELS[1]], columns=[LABELS[-1], LABELS[0], LABELS[1]])
            overall_df.to_csv(outdir / f'direction_confusion_overall_h{h}.csv')

    # Save per-company confusion matrices (stacked)
    if per_company_cm_rows:
        stacked = pd.concat(per_company_cm_rows, ignore_index=True)
        stacked.to_csv(outdir / 'direction_confusion_per_company.csv', index=False)

    # Save metrics
    if all_metrics:
        metrics = pd.concat(all_metrics, ignore_index=True)
        metrics.sort_values(['horizon','company'], inplace=True)
        metrics.to_csv(outdir / 'direction_classification_metrics.csv', index=False)

    # README for reproducibility
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
            f'Test size (proportion): {cfg.test_size}\n'
            'Labels: -1=down, 0=flat, 1=up. Confusion matrices are row-normalized by true class.\n'
        )


# Example (not executed here):
# from pathlib import Path
# df = pd.read_csv('your_prices.csv')
# run_direction_pipeline(df, Path('outputs'), horizons=(1,2,3,4,5), n_lags=10, flat_threshold_pct=0.2)
