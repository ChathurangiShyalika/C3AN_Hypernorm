Direction classification pipeline
Model type: logistic
Use technical indicators: True
Aggregate daily: True
Horizons: [1, 2, 3, 4, 5]
Lags: 10
Price column: ohlc_avg
Flat threshold (%): 0.2
Test size (proportion): 0.2
Labels: -1=down, 0=flat, 1=up. Confusion matrices are row-normalized by true class.
