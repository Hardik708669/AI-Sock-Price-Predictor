import pandas as pd


class MarketTrendDetector:
    def detect(self, data: pd.DataFrame) -> dict:
        close = data["Close"]
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        returns_90d = close.pct_change(90)
        latest_price = float(close.iloc[-1])
        latest_sma_50 = float(sma_50.dropna().iloc[-1])
        latest_sma_200 = float(sma_200.dropna().iloc[-1]) if sma_200.notna().any() else latest_sma_50
        momentum = float(returns_90d.dropna().iloc[-1]) if returns_90d.notna().any() else 0.0
        if latest_price > latest_sma_50 > latest_sma_200 and momentum > 0.05:
            regime = "Bull Market"
        elif latest_price < latest_sma_50 < latest_sma_200 and momentum < -0.05:
            regime = "Bear Market"
        else:
            regime = "Sideways Market"
        return {"regime": regime, "momentum_90d": round(momentum, 4), "price_vs_sma50": round((latest_price / latest_sma_50) - 1, 4), "sma50_vs_sma200": round((latest_sma_50 / latest_sma_200) - 1, 4)}
