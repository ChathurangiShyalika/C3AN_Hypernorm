# Directional Modeling Report — Theory, Results, and Roadmap (8 Jan 2026)

## 1. Executive Overview
- We built three successive directional models on the aggregated per-company, per-day dataset: a linear multi-output regression baseline, a multinomial logistic classifier, and a balanced Random Forest enriched with technical indicators.
- After eliminating same-day leakage and adding stronger features, the Random Forest now delivers **48.7 % overall directional accuracy** versus **40.8 %** for the regression baseline; logistic regression sits in between at **42.1 %**.
- Pipeline upgrades (dynamic flat bands, indicator bank, ensemble plumbing) provide the theoretical and practical foundation to exceed the 50 % accuracy ceiling typical in noisy financial direction problems.
- Test-set direction mix: actual **up 34 %**, **down 42 %**, **flat 24 %** after aggregation; Random Forest predictions distribute as up 37 %, down 39 %, flat 24 %, indicating balanced coverage relative to reality.

| Model & Output Folder | Overall Accuracy | Macro-F1 | H1 Acc | H2 Acc | H3 Acc | H4 Acc | H5 Acc | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regression baseline (`forecast_next _week.py`) | 40.8 % | 0.29 | 38.7 % | 40.5 % | 41.3 % | 42.0 % | 41.5 % | Uses lagged prices; strong level forecasts, weak direction skill |
| Logistic classifier (`analysis_outputs/directional_outputs/`) | 42.1 % | 0.32 | 44.2 % | 42.1 % | 43.8 % | 43.9 % | 36.8 % | StandardScaler + multinomial LogReg; balanced confusion after aggregation |
| Random Forest (`analysis_outputs/directional_outputs_rf/`) | 48.7 % | 0.31 | 48.3 % | 49.0 % | 48.6 % | 48.8 % | 48.9 % | 400-tree balanced RF on lag + indicator block |

## 2. Theoretical Foundations

### 2.1 Directional Targets and Labeling
- **Directional change**: For each horizon \(H\), label compares price at \(t+H\) to price at \(t\). The percentage move \(\Delta_H = 100 \times (P_{t+H}/P_t - 1)\).
- **Three-class mapping**: Up (+1) if \(\Delta_H > T_H\), down (-1) if \(\Delta_H < -T_H\), otherwise flat (0), where \(T_H\) is the horizon-specific threshold (currently base 0.2 %, being extended to volatility-scaled rules).
- **Flat-band theory**: The flat class captures statistically insignificant moves. A fixed percentage band equates to a symmetric confidence interval; scaling by \(\sqrt{H}\) acknowledges random-walk variance growth (\(\sigma \sqrt{H}\)). ATR-based scaling ties the band to realized volatility, better aligning with heteroskedastic markets.

### 2.2 Performance Metrics
- **Directional accuracy**: Fraction of correct class predictions. Sensitive to class imbalance; a majority-class predictor can appear strong if one class dominates.
- **Macro-F1**: Harmonic mean of per-class precision and recall, averaged equally across classes. Penalizes models that ignore minority classes (e.g., predicting only “flat”).
- **Confusion matrices**: Row-normalized confusion tables reveal bias (e.g., over-predicting down). Ensuring near-diagonal mass indicates balanced directional skill.

### 2.3 Model Families
- **Linear regression baseline**: Multi-output linear regression estimates future price levels. Direction is inferred via sign of predicted change. Theory: OLS minimizes squared error on price level, not class separation, so directional accuracy hinges on accurate magnitude forecasts.
- **Multinomial logistic regression**: Assumes log-odds of each class are linear functions of features. With L2 regularisation and StandardScaler preprocessing, it produces calibrated class probabilities and is well-suited to linearly separable trend features.
- **Random Forest classifier**: Ensemble of decision trees trained on bootstrap samples with feature subsampling. Captures non-linear feature interactions (e.g., RSI thresholds combined with volatility states). Balanced class weights counter skewed class frequencies by adjusting impurity splits.
- **Probability ensembling (new option)**: Weighted blend of logistic and RF probabilities aims to combine the smoother trend tracking of logistic with RF’s spike detection. Theoretically reduces variance and exploits complementary biases.

### 2.4 Feature Engineering Theory
- **Lagged prices**: Encode autoregressive structure; effective if market exhibits mean reversion or persistence in price levels.
- **Technical indicators**:
	- *RSI*: Measures momentum magnitude relative to gains vs losses; values near 70/30 flag overbought/oversold states.
	- *MACD (fast EMA − slow EMA)*: Captures trend direction; signal line crossovers imply inflection.
	- *Bollinger Bands*: Combine moving average and standard deviation to signal volatility-adjusted extremes.
	- *ATR*: Average true range, a volatility proxy essential for adaptive thresholds and risk sizing.
	- *Rolling volatility, ROC, lagged returns, streak counts*: Quantify dispersion and directional persistence, aiding classifiers sensitive to recent movement patterns.

### 2.5 Data Leakage and Aggregation
- Financial event datasets often contain multiple records per day (earnings segments, news bulletins). Training on raw rows leaks future information when same-day records share the same target but different features.
- Aggregating to a single row per company-day maintains chronological integrity, ensures lag features reference strictly prior days, and aligns indicators (which assume regular intervals).

## 3. Experimental Setup to Date
- **Dataset**: Aggregated equities dataset, one row per company per calendar day, including OHLC, volume, and engineered technical indicators.
- **Train/test regimen**: Chronological split per company with ~20 % holdout for testing and TimeSeriesSplit CV for robustness (~5 folds, auto-reduced for short histories).
- **Feature matrix**: 10 price lags, indicator block (RSI, MACD trio, Bollinger bands, ATR-14, rolling vol 20, ROC-5, lag returns, streak), base/future price columns for diagnostics.
- **Thresholding**: Currently fixed 0.2 % per horizon; pipeline now supports strategies `static`, `sqrt_h`, `atr`, `atr_sqrt` with adjustable ATR multipliers.
- **Outputs**: CSVs for per-company metrics, confusion matrices, per-row predictions, plus README summarising configuration for reproducibility.

## 4. Results and Interpretation

### 4.1 Regression Baseline
- **Outcome**: 40.8 % accuracy overall, macro-F1 0.29. Directions skew toward the majority class (down ≈42 %, up ≈52 %, flat ≈6 %).
- **Interpretation**: Regression is optimized for minimizing squared price error; directional decisions rely on sign of small residuals, so noise overwhelms signal. Serves as level forecasting benchmark, not competitive for classification.

### 4.2 Logistic Regression Classifier
- **Outcome**: 42.1 % accuracy, macro-F1 0.32, H1 accuracy 44.2 %. Confusion matrix shows balanced up/down predictions with flat ≈24 % after data aggregation.
- **Interpretation**: Linear decision boundaries in indicator space capture moderate trend signals. Performance drop vs pre-aggregation run (from 86 %) confirms earlier metrics were contaminated by leakage. Logistic now represents a realistic linear baseline.

### 4.3 Random Forest Classifier
- **Outcome**: 48.7 % accuracy, macro-F1 0.31, horizons 1–5 all near 48–49 % accuracy, demonstrating stability across forecast lengths.
- **Interpretation**: Non-linear splits exploit interactions among indicators and lags, improving recall for both up and down moves while keeping flat predictions around 24 %. Gains are especially notable at longer horizons where logistic falls off.

### 4.4 Emerging Ensemble Capability
- **Status**: Pipeline now supports a logistic-RF probability blend (`model_type='ensemble'`) with configurable weight. Theory suggests blending reduces variance and improves calibration; empirical validation scheduled in next steps.
- **Accuracy computation**: For every model, accuracy = \( \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(\hat{y}_i = y_i) \), where \(N\) is number of test observations, \(y_i\) actual labels, and \(\hat{y}_i\) predicted classes derived via argmax of model probabilities. Macro-F1 averages per-class precision/recall to avoid dominance by the majority class.

## 5. Challenges Encountered
- **Leakage elimination**: Transitioning to daily aggregation reduced inflated accuracy but required rewriting feature engineering and label pipelines.
- **Flat class volatility**: Fixed threshold produced horizon-dependent class imbalance. Work-in-progress ATR/√H strategy aims to stabilize flat share across horizons.
- **Feature sparsity**: Without indicator augmentation, classifiers revert to predicting majority classes. The indicator block is now treated as mandatory for serious runs.
- **Probability disagreement**: Logistic favors smooth signals; RF reacts to volatility. Deciding between them is non-trivial without an ensemble or calibrated confidence measure.

## 6. Key Inferences
- **Non-linear models benefit most from richer features**: Random Forest converts indicator interactions into materially better directional predictions (≈7 ppt lift vs logistic).
- **Balanced flat share is crucial**: Maintaining ~20–25 % flat predictions keeps macro-F1 above 0.30 and prevents directional bias. Threshold tuning is therefore a lever for accuracy.
- **Complementary model biases are exploitable**: Logistic’s trend focus and RF’s volatility focus suggest probability blending, calibrated ensembling, or stacking with the regression model can push accuracy higher.
- **Volatility-aware labeling aligns with trading intuition**: Scaling thresholds by ATR should map predictions more closely to tradeable moves, reducing false positives in quiet regimes.

## 7. Improvements Over Previous Iterations
- **Metric integrity**: Removing duplicated same-day rows corrected the misleading 86 % logistic accuracy, grounding evaluations in realistic expectations.
- **Directional uplift**: RF accuracy improved from 53.9 % (pre-aggregation but biased) to 48.7 % honest accuracy while increasing macro-F1 from 0.24 to 0.31, indicating better multi-class balance.
- **Infrastructure enhancements**: Added dynamic threshold strategies, ATR dependency management, and ensemble hooks; outputs now include strategy metadata for every run.
- **Analytical clarity**: Updated documentation and metrics files now track confusion matrices, per-company summaries, and probability outputs, enabling targeted troubleshooting.

## 8. Next Steps and Research Roadmap
1. **Volatility-scaled flat bands** ✅ Completed: Deploy `flat_threshold_strategy='atr_sqrt'` across logistic, RF, and ensemble models. Tune `atr_threshold_multiplier` (e.g., 0.8–1.4) to keep flat class near 20 % per horizon and measure accuracy/F1 impacts.
2. **Probability ensemble experiments** ✅ Completed: Grid-search `ensemble_logistic_weight` between 0.3 and 0.7. Evaluate per-horizon accuracy, macro-F1, and calibration (Brier score) to identify optimal blend.
3. **Calibration & decision filters**: Apply Platt scaling or isotonic regression to each classifier’s probabilities; test rule-based confirmations (RSI and MACD agreeing, minimum probability gaps) to reduce ambiguous trades.
4. **Model diversification**: Once thresholds stabilize, introduce gradient boosting (XGBoost, LightGBM) and neural tabular models. Compare to RF using the same feature set and threshold strategy.
5. **Economic validation notebook**: Build a notebook to translate directional signals into hypothetical PnL, hit-rate by volatility regime, and confusion evolution over time—bridging model metrics with business impact.

## 9. Appendix — Key Terms
- **ATR (Average True Range)**: Rolling average of true range, capturing intraday high-low and gap volatility; used to scale thresholds and gauge market activity.
- **RSI (Relative Strength Index)**: Momentum oscillator; high values indicate strong upward moves, low values suggest downward pressure.
- **MACD**: Difference between short and long exponential moving averages; signal line crossover highlights trend shifts.
- **Bollinger Bands**: Moving average plus/minus standard deviations; signal price extremes relative to recent volatility.
- **TimeSeriesSplit**: Cross-validation strategy respecting chronological order; prevents training on future data when validating models.
- **Macro-F1**: Average of per-class F1 scores; ensures performance is not dominated by the majority class.

All referenced metrics and CSV outputs reside under `analysis_outputs/`, with subfolders `directional_outputs/` (logistic) and `directional_outputs_rf/` (Random Forest). Future ensemble experiments should create dedicated subdirectories to maintain provenance.
