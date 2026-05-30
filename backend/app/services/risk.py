import numpy as np

from app.services.market_data import MarketDataService


class RiskService:
    def __init__(self, market_data: MarketDataService):
        self.market_data = market_data

    def analyze(self, symbol: str) -> dict:
        stock = self.market_data.history(symbol, period="2y")["Close"].pct_change().dropna()
        benchmark = self.market_data.history("^GSPC", period="2y")["Close"].pct_change().dropna()
        aligned = stock.align(benchmark, join="inner")
        stock_returns, benchmark_returns = aligned
        volatility = float(stock_returns.std() * np.sqrt(252))
        sharpe = float((stock_returns.mean() * 252) / (stock_returns.std() * np.sqrt(252)))
        cumulative = (1 + stock_returns).cumprod()
        drawdown = float(((cumulative / cumulative.cummax()) - 1).min())
        beta = float(np.cov(stock_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns))
        risk_label = "Low Risk" if volatility < 0.2 else "Medium Risk" if volatility < 0.38 else "High Risk"
        return {
            "symbol": symbol.upper(),
            "volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe, 4),
            "maximum_drawdown": round(drawdown, 4),
            "beta": round(beta, 4),
            "risk_level": risk_label,
        }
