Direction classification pipeline
Model type: ensemble
Use technical indicators: True
Aggregate daily: True
Horizons: [1, 2, 3, 4, 5]
Lags: 10
Price column: ohlc_avg
Flat threshold (%): 0.1
Flat threshold strategy: atr_sqrt
ATR threshold multiplier: 0.5
Ensemble logistic weight: 0.5
Prediction margin threshold: 0.1
Log run metadata: True
Test size (proportion): 0.2
Labels: -1=down, 0=flat, 1=up. Confusion matrices are row-normalized by true class.
