from pathlib import Path
import argparse
import pandas as pd
from direction_pipeline import run_direction_pipeline


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the directional forecasting pipeline.")
    parser.add_argument(
        "--model-type",
        default="ensemble",
        choices=["logistic", "random_forest", "ensemble", "xgboost", "tcn"],
        help="Which classifier to train. TCN requires PyTorch to be installed.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the output directory.",
    )
    parser.add_argument(
        "--n-lags",
        type=int,
        default=None,
        help="Override the number of lag features (default 10, 30 for TCN).",
    )
    parser.add_argument("--tcn-hidden-channels", type=int, default=None, help="Hidden channels for the TCN backbone.")
    parser.add_argument("--tcn-num-layers", type=int, default=None, help="Number of temporal convolution blocks.")
    parser.add_argument("--tcn-kernel-size", type=int, default=None, help="Kernel size for temporal convolutions.")
    parser.add_argument("--tcn-dropout", type=float, default=None, help="Dropout applied after each temporal block.")
    parser.add_argument("--tcn-lr", type=float, default=None, help="Learning rate for the TCN optimizer.")
    parser.add_argument("--tcn-batch-size", type=int, default=None, help="Batch size for TCN training.")
    parser.add_argument("--tcn-max-epochs", type=int, default=None, help="Maximum training epochs for the TCN.")
    parser.add_argument("--tcn-patience", type=int, default=None, help="Early-stopping patience for the TCN.")
    parser.add_argument(
        "--tcn-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for TCN training: auto, cuda, or cpu.",
    )
    return parser.parse_args()


def _default_output_dir(model_type: str) -> str:
    base = "analysis_outputs/final_direction_model"
    model_type = (model_type or "").lower()
    if model_type in {"", "ensemble"}:
        return base
    return f"{base}_{model_type}"

def main():
    args = _parse_args()
    DATA_FILE = "asset_events_df.csv"
    OUTPUT_DIR = args.output_dir or _default_output_dir(args.model_type)

    is_tcn = args.model_type == "tcn"
    default_n_lags = 30 if is_tcn else 10
    effective_n_lags = args.n_lags if args.n_lags is not None else default_n_lags

    tcn_defaults = dict(
        hidden_channels=128,
        num_layers=4,
        kernel_size=5,
        dropout=0.2,
        lr=5e-4,
        batch_size=128,
        max_epochs=80,
        patience=10,
    )

    def _tcn_value(arg_name: str, key: str):
        val = getattr(args, arg_name)
        return val if val is not None else tcn_defaults[key]

    tcn_kwargs = dict(
        tcn_hidden_channels=_tcn_value("tcn_hidden_channels", "hidden_channels") if is_tcn else 64,
        tcn_num_layers=_tcn_value("tcn_num_layers", "num_layers") if is_tcn else 3,
        tcn_kernel_size=_tcn_value("tcn_kernel_size", "kernel_size") if is_tcn else 3,
        tcn_dropout=_tcn_value("tcn_dropout", "dropout") if is_tcn else 0.1,
        tcn_lr=_tcn_value("tcn_lr", "lr") if is_tcn else 1e-3,
        tcn_batch_size=_tcn_value("tcn_batch_size", "batch_size") if is_tcn else 256,
        tcn_max_epochs=_tcn_value("tcn_max_epochs", "max_epochs") if is_tcn else 50,
        tcn_patience=_tcn_value("tcn_patience", "patience") if is_tcn else 6,
        tcn_device=args.tcn_device if is_tcn else "auto",
    )
    
    pipeline_kwargs = dict(
        horizons=(1, 2, 3, 4, 5),
        n_lags=effective_n_lags,
        
        # Threshold Strategy: Tight ATR limits to distinguish real moves from noise
        flat_threshold_strategy="atr_sqrt",
        flat_threshold_pct=0.1,          # Base threshold
        atr_threshold_multiplier=0.5,    # kept for compatibility; up/down override it
        atr_up_multiplier=0.6,           # Best-performing Up threshold per sweep
        atr_down_multiplier=0.5,         # Best-performing Down threshold per sweep
        prediction_margin_threshold=0.15, # Require 15% probability margin for operability tagging
        
        # Feature Engineering
        use_technical_indicators=True,    # Includes RSI, MACD, BB, and Relative Volume
        aggregate_daily=True,             # Ensure one row per company/day
        
        # Model Configuration
        model_type=args.model_type,
        ensemble_logistic_weight=0.5,     # Equal weight to linear and tree models
        
        # Training Validation
        balance_classes=True,
        test_size=0.2,
        n_splits_cv=5,
        **tcn_kwargs,
    )

    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=["date"])
        df["company"] = df["company"].astype("category")
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_FILE}. Please ensure it is in the root directory.")
        return

    print(f"Starting model run with model_type='{args.model_type}' (ATR thresholds + vol features)...")
    print(f"Saving outputs to: {OUTPUT_DIR}")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    run_direction_pipeline(
        df, 
        OUTPUT_DIR, 
        **pipeline_kwargs
    )
    
    print("\nRun Complete.")
    print(f"Check {OUTPUT_DIR}/direction_classification_metrics.csv for results.")

if __name__ == "__main__":
    main()
