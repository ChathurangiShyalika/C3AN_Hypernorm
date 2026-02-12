from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

from direction_pipeline import run_direction_pipeline

FLAT_THRESHOLDS = [0.05, 0.1, 0.15]
ATR_UP_MULTS = [0.4, 0.5, 0.6]
ATR_DOWN_OFFSETS = [-0.1, -0.1, -0.1]  # down multiplier = up + offset (offset negative)

OUTPUT_ROOT = Path("analysis_outputs/threshold_sweep")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = dict(
    horizons=(1, 2, 3, 4, 5),
    n_lags=10,
    flat_threshold_strategy="atr_sqrt",
    atr_threshold_multiplier=0.5,
    use_technical_indicators=True,
    aggregate_daily=True,
    model_type="ensemble",
    ensemble_logistic_weight=0.5,
    balance_classes=True,
    test_size=0.2,
    n_splits_cv=5,
    prediction_margin_threshold=0.0,
    log_run_metadata=False,
)


def _iter_configs() -> Iterable[dict]:
    for flat_pct in FLAT_THRESHOLDS:
        for up_mult, offset in zip(ATR_UP_MULTS, ATR_DOWN_OFFSETS):
            down_mult = max(0.1, up_mult + offset)
            yield {
                'flat_threshold_pct': flat_pct,
                'atr_up_multiplier': up_mult,
                'atr_down_multiplier': down_mult,
            }


def main() -> None:
    data_file = Path('asset_events_df.csv')
    df = pd.read_csv(data_file, parse_dates=['date'])
    df['company'] = df['company'].astype('category')

    summary_rows = []

    for cfg in _iter_configs():
        label = f"flat_{cfg['flat_threshold_pct']:.2f}_up_{cfg['atr_up_multiplier']:.2f}_down_{cfg['atr_down_multiplier']:.2f}"
        label = label.replace('.', 'p')
        outdir = OUTPUT_ROOT / label
        outdir.mkdir(parents=True, exist_ok=True)

        run_direction_pipeline(
            df.copy(),
            outdir,
            **BASE_CONFIG,
            **cfg,
        )

        metrics_path = outdir / 'direction_classification_metrics.csv'
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        agg = metrics.groupby('horizon')[['acc_test', 'f1_macro_test']].mean().reset_index()
        agg['flat_threshold_pct'] = cfg['flat_threshold_pct']
        agg['atr_up_multiplier'] = cfg['atr_up_multiplier']
        agg['atr_down_multiplier'] = cfg['atr_down_multiplier']
        summary_rows.append(agg)

    if summary_rows:
        summary_df = pd.concat(summary_rows, ignore_index=True)
        summary_df.to_csv(OUTPUT_ROOT / 'threshold_sweep_summary.csv', index=False)
        print('Sweep complete. Summary at', OUTPUT_ROOT / 'threshold_sweep_summary.csv')
    else:
        print('No results produced.')


if __name__ == '__main__':
    main()
