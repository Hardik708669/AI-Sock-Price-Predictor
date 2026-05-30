from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

from app.schemas import Candle, StockOverview, TechnicalSignals


DEFAULT_SYMBOLS = ["AAPL", "TSLA", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS"]


class MarketDataService:
    def search(self, query: str) -> list[dict]:
        query = query.upper().strip()
        universe = [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "sector": "Technology"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ", "sector": "Consumer Cyclical"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "sector": "Technology"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "sector": "Technology"},
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "exchange": "NSE", "sector": "Energy"},
            {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "exchange": "NSE", "sector": "Technology"},
        ]
        return [item for item in universe if query in item["symbol"] or query in item["name"].upper()][:8]

    def history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if data.empty:
            raise ValueError(f"No market data available for {symbol}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna()

    def candles(self, symbol: str, period: str = "1y") -> list[Candle]:
        data = self.history(symbol, period=period)
        return [
            Candle(
                date=index.strftime("%Y-%m-%d"),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
            for index, row in data.tail(260).iterrows()
        ]

    def overview(self, symbol: str) -> StockOverview:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        data = self.history(symbol, period="6mo")
        live_price = float(data["Close"].iloc[-1])
        return StockOverview(
            symbol=symbol.upper(),
            name=info.get("longName") or info.get("shortName") or symbol.upper(),
            live_price=live_price,
            currency=info.get("currency") or "USD",
            market_cap=info.get("marketCap"),
            volume=float(data["Volume"].iloc[-1]),
            pe_ratio=info.get("trailingPE"),
            eps=info.get("trailingEps"),
            dividend_yield=info.get("dividendYield"),
            sector=info.get("sector") or "Unknown",
            summary=(info.get("longBusinessSummary") or "Company profile unavailable.")[:700],
        )

    def technicals(self, symbol: str) -> TechnicalSignals:
        data = self.history(symbol, period="1y")
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return TechnicalSignals(
            rsi=float(rsi.iloc[-1]),
            macd=float(macd.iloc[-1]),
            ema_20=float(close.ewm(span=20, adjust=False).mean().iloc[-1]),
            sma_50=float(close.rolling(50).mean().iloc[-1]),
            bollinger_upper=float((sma_20 + 2 * std_20).iloc[-1]),
            bollinger_lower=float((sma_20 - 2 * std_20).iloc[-1]),
            vwap=float(vwap.iloc[-1]),
        )


@lru_cache
def get_market_data_service() -> MarketDataService:
    return MarketDataService()
