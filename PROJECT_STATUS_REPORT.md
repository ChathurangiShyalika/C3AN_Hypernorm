# Project Status Report: Directional Forecasting Model

## Executive Summary
The directional ensemble (logistic + random forest) is now locked with the asymmetric ATR thresholds discovered in the recent sweep (flat threshold 10%, ATR up 0.6, ATR down 0.5). This configuration lifts Horizon-1 accuracy to 64% while keeping Horizons 2-5 near 55-60%, providing a firmer numeric baseline before adding macro/text signals. Confidence gating (δ) now uses the same dir-vs-flat operability logic that matches the paper, and the fresh sweep quantifies the trade-off between coverage and win rate.

### Key Improvements
1. **Dynamic ATR Thresholds**: Swept `flat_threshold_pct` × ATR multipliers and standardized on flat=0.10, up=0.60, down=0.50 after the configuration delivered the best cross-horizon accuracy ([analysis_outputs/threshold_sweep/threshold_sweep_summary.csv](analysis_outputs/threshold_sweep/threshold_sweep_summary.csv)).
2. **Pipeline Corrections**: Persisted the cleaned label mapping and balancing logic that fixed the earlier XGBoost label inversion bug.
3. **Feature Engineering**: Retained the volatility + volume feature set that stabilizes the ensemble (relative volume, log volume change, ATR, RSI, etc.).
4. **Ensemble Architecture**: The logistic head continues to temper the RF logits, preventing mode collapse into all-flat predictions.
5. **Confidence Gating Alignment**: Rebuilt the operability definition so that δ compares directional probability vs flat; the latest sweep quantifies where to place the production δ.

## Current Performance State (Baseline Run 2026-02-05)
Source files: [analysis_outputs/final_direction_model/run_summary.json](analysis_outputs/final_direction_model/run_summary.json) and the associated confusion matrices/prediction dumps under the same folder. The table below summarizes accuracy, macro F1, and operability at δ = 0.15.

| Horizon | Accuracy | Macro F1 | Operable Share (δ=0.15) | Operable Accuracy |
| :-- | --: | --: | --: | --: |
| 1 | 64.4% | 33.5% | 11.9% | 26.0% |
| 2 | 59.5% | 29.8% | 8.8% | 27.8% |
| 3 | 58.0% | 30.9% | 10.8% | 28.7% |
| 4 | 56.7% | 31.2% | 12.5% | 29.7% |
| 5 | 54.9% | 31.2% | 15.4% | 28.1% |

Notes:
- Totals per horizon remain 4,936 samples (20% test split) spanning seven mega-cap tickers.
- Company-level breakouts live in [analysis_outputs/final_direction_model/direction_classification_metrics.csv](analysis_outputs/final_direction_model/direction_classification_metrics.csv), with Alphabet/Meta leading on near-term accuracy and NVIDIA lagging.
- Feature importance shifts remain shallow, suggesting macro/text integration will have clear headroom.

## Operability (δ) Sweep Insights
Latest sweep reran all horizons for δ ∈ {0.00, 0.02, 0.05, 0.08, 0.10, 0.15} using the new ATR multipliers. Results: [analysis_outputs/operability_sweep/operability_sweep_summary.csv](analysis_outputs/operability_sweep/operability_sweep_summary.csv) and [analysis_outputs/operability_sweep/operability_margin_distribution.csv](analysis_outputs/operability_sweep/operability_margin_distribution.csv).

| δ | H1 Operable Share | H1 Operable Accuracy | H1 Directional Recall Avg |
| :-- | --: | --: | --: |
| 0.00 | 29.1% | 25.5% | 23.4% |
| 0.05 | 21.5% | 26.6% | 18.0% |
| 0.15 | 11.9% | 26.0% | 9.8% |

- Lower δ dramatically expands coverage but only nudges accuracy; δ=0 already surfaces 29% of cases with ~25% win rate, which may be attractive for research notebooks but not production.
- Higher δ slashes recall (directional trades triggered) yet leaves win rate roughly flat; the sweet spot likely sits between 0.05 and 0.10 if we need more tickets without sacrificing precision.
- Margin quantiles barely move across δ because they reflect the raw probability separation; see the margin distribution CSV for exact values.

## Next Steps
1. Port the macro + text features (macro panel plus earnings call-derived sentiment) into the same pipeline now that the numeric baseline is stable.
2. Re-fit the δ policy after macro/text go in; we expect the directional margin curve to steepen, so δ=0.10 may become viable.
3. Promote the current config to the experimentation registry so future sweeps inherit the locked ATR and flat threshold values.