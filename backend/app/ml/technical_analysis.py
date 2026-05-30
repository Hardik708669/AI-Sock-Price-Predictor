import numpy as np
import pandas as pd

from app.ml.features import average_true_range
from app.ml.optional import optional_import


class TechnicalAnalysisEngine:
    def indicators(self, data: pd.DataFrame) -> dict:
        talib = optional_import("talib")
        close = data["Close"].astype(float)
        high = data["High"].astype(float)
        low = data["Low"].astype(float)
        volume = data["Volume"].astype(float)

        if talib:
            rsi = pd.Series(talib.RSI(close.to_numpy(), timeperiod=14), index=data.index)
            macd, macd_signal, macd_hist = talib.MACD(close.to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9)
            atr = pd.Series(talib.ATR(high.to_numpy(), low.to_numpy(), close.to_numpy(), timeperiod=14), index=data.index)
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_signal
            atr = average_true_range(data, 14)

        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_20 = close.ewm(span=20, adjust=False).mean()
        std_20 = close.rolling(20).std()
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.replace(0, np.nan).cumsum()

        return {
            "rsi": float(rsi.dropna().iloc[-1]),
            "macd": float(pd.Series(macd, index=data.index).dropna().iloc[-1]),
            "macd_signal": float(pd.Series(macd_signal, index=data.index).dropna().iloc[-1]),
            "macd_histogram": float(pd.Series(macd_hist, index=data.index).dropna().iloc[-1]),
            "sma_20": float(sma_20.dropna().iloc[-1]),
            "sma_50": float(sma_50.dropna().iloc[-1]),
            "ema_20": float(ema_20.dropna().iloc[-1]),
            "bollinger_upper": float((sma_20 + 2 * std_20).dropna().iloc[-1]),
            "bollinger_lower": float((sma_20 - 2 * std_20).dropna().iloc[-1]),
            "vwap": float(vwap.dropna().iloc[-1]),
            "atr": float(atr.dropna().iloc[-1]),
        }

    def signal(self, data: pd.DataFrame) -> dict:
        indicators = self.indicators(data)
        price = float(data["Close"].iloc[-1])
        score = 0
        reasons: list[str] = []
        if indicators["rsi"] < 35:
            score += 2
            reasons.append("RSI is oversold")
        elif indicators["rsi"] > 70:
            score -= 2
            reasons.append("RSI is overbought")
        if indicators["macd"] > indicators["macd_signal"]:
            score += 1
            reasons.append("MACD is above signal")
        else:
            score -= 1
            reasons.append("MACD is below signal")
        if price > indicators["sma_20"] > indicators["sma_50"]:
            score += 2
            reasons.append("Price is in bullish moving-average alignment")
        elif price < indicators["sma_20"] < indicators["sma_50"]:
            score -= 2
            reasons.append("Price is in bearish moving-average alignment")
        if price <= indicators["bollinger_lower"]:
            score += 1
            reasons.append("Price is near lower Bollinger Band")
        if price >= indicators["bollinger_upper"]:
            score -= 1
            reasons.append("Price is near upper Bollinger Band")
        action = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
        return {"signal": action, "score": score, "reasons": reasons, "indicators": indicators}
