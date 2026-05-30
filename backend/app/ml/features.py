import pandas as pd


def technical_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    close = data["Close"]
    return pd.DataFrame(
        {
            "return_1d": close.pct_change(),
            "return_5d": close.pct_change(5),
            "sma_10": close.rolling(10).mean(),
            "sma_30": close.rolling(30).mean(),
            "volatility_20": close.pct_change().rolling(20).std(),
            "volume_change": data["Volume"].pct_change(),
        }
    ).dropna()


def engineer_features(data: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]
    returns = close.pct_change()
    features = pd.DataFrame(index=data.index)
    features["return_1d"] = returns
    features["return_5d"] = close.pct_change(5)
    features["return_21d"] = close.pct_change(21)
    features["sma_10"] = close.rolling(10).mean()
    features["sma_20"] = close.rolling(20).mean()
    features["sma_50"] = close.rolling(50).mean()
    features["ema_12"] = close.ewm(span=12, adjust=False).mean()
    features["ema_26"] = close.ewm(span=26, adjust=False).mean()
    features["macd"] = features["ema_12"] - features["ema_26"]
    features["volatility_10"] = returns.rolling(10).std()
    features["volatility_21"] = returns.rolling(21).std()
    features["volume_change"] = volume.pct_change()
    features["volume_zscore"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    features["high_low_range"] = (high - low) / close
    features["atr_14"] = average_true_range(data, 14)
    for lag in [1, 2, 3, 5, 10]:
        features[f"close_lag_{lag}"] = close.shift(lag)
        features[f"return_lag_{lag}"] = returns.shift(lag)
    target = close.shift(-horizon).rename("target")
    dataset = pd.concat([features, target], axis=1).replace([float("inf"), float("-inf")], pd.NA).dropna()
    return dataset.drop(columns=["target"]), dataset["target"]


def average_true_range(data: pd.DataFrame, window: int = 14) -> pd.Series:
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    previous_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(window).mean()
