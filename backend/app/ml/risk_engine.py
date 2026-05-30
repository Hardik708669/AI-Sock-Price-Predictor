import numpy as np
import pandas as pd


class RiskAnalysisEngine:
    def analyze(self, data: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> dict:
        returns = data["Close"].pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252))
        downside = returns[returns < 0]
        sharpe = float((returns.mean() * 252) / max(returns.std() * np.sqrt(252), 1e-9))
        sortino = float((returns.mean() * 252) / max(downside.std() * np.sqrt(252), 1e-9)) if len(downside) else sharpe
        cumulative = (1 + returns).cumprod()
        max_drawdown = float(((cumulative / cumulative.cummax()) - 1).min())
        beta = 1.0
        if benchmark is not None:
            benchmark_returns = benchmark["Close"].pct_change().dropna()
            aligned_stock, aligned_benchmark = returns.align(benchmark_returns, join="inner")
            if len(aligned_stock) > 5 and float(np.var(aligned_benchmark)) != 0:
                beta = float(np.cov(aligned_stock, aligned_benchmark)[0, 1] / np.var(aligned_benchmark))
        risk_score = min(100, max(0, volatility * 90 + abs(max_drawdown) * 80 + max(beta - 1, 0) * 20 - max(sharpe, 0) * 8))
        level = "Low" if risk_score < 35 else "Medium" if risk_score < 68 else "High"
        return {
            "volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "beta": round(beta, 4),
            "maximum_drawdown": round(max_drawdown, 4),
            "risk_score": round(float(risk_score), 2),
            "risk_level": level,
        }
