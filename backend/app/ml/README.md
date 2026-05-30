# StockVision AI Engine

The AI engine is organized as independent modules under `backend/app/ml`.

## Pipeline

Data collection -> cleaning -> feature engineering -> training -> validation -> testing -> registry/deployment.

## Modules

- `data_pipeline.py`: Yahoo Finance, optional Alpha Vantage, optional financial news API, OHLCV cleaning.
- `features.py`: moving averages, momentum, volume indicators, lag features, rolling statistics, ATR.
- `prediction_engine.py`: Linear Regression, Random Forest, XGBoost, optional Prophet, optional TensorFlow LSTM.
- `technical_analysis.py`: RSI, MACD, SMA, EMA, Bollinger Bands, VWAP, ATR, BUY/SELL/HOLD signals.
- `sentiment_engine.py`: optional FinBERT transformers pipeline with lexicon fallback.
- `trend_detection.py`: bull, bear, and sideways market regime detection.
- `risk_engine.py`: volatility, Sharpe, Sortino, beta, maximum drawdown, risk score.
- `optimizer.py`: risk-appetite portfolio allocation.
- `explainability.py`: SHAP feature importance with model-native fallback.
- `anomaly_detection.py`: price spike, abnormal volume, and manipulation-style anomaly flags.
- `forecasting.py`: 1, 7, 30, and 90 day forecasts.
- `copilot.py`: combines prediction, technicals, sentiment, risk, trend, and anomalies.
- `registry.py`: model versioning and metadata registry.
- `training_pipeline.py`: retraining and registration pipeline.
- `engine.py`: high-level orchestrator used by API routes.

## API

All routes require JWT authentication.

- `GET /api/v1/ai/intelligence/{ticker}`
- `GET /api/v1/ai/technical-analysis/{ticker}`
- `GET /api/v1/ai/sentiment/{ticker}`
- `GET /api/v1/ai/trend/{ticker}`
- `GET /api/v1/ai/risk/{ticker}`
- `GET /api/v1/ai/anomalies/{ticker}`
- `GET /api/v1/ai/forecast/{ticker}`
- `POST /api/v1/ai/portfolio/optimize`
- `POST /api/v1/ai/retrain`

## Optional Native Dependencies

Prophet, TensorFlow, TA-Lib, SHAP, and Transformers are used when installed. The engine keeps deterministic fallbacks so local development remains usable on machines where native wheels are unavailable.
