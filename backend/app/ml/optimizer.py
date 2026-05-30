import numpy as np
import pandas as pd


class PortfolioOptimizer:
    risk_targets = {"conservative": 0.08, "moderate": 0.14, "aggressive": 0.22}

    def optimize(self, price_history: dict[str, pd.DataFrame], investment_amount: float, risk_appetite: str = "moderate") -> dict:
        if investment_amount <= 0:
            raise ValueError("investment_amount must be greater than zero")
        returns = pd.DataFrame({ticker: frame["Close"].pct_change() for ticker, frame in price_history.items()}).dropna()
        if returns.empty:
            raise ValueError("Not enough historical data to optimize portfolio")
        expected_returns = returns.mean() * 252
        covariance = returns.cov() * 252
        target_volatility = self.risk_targets.get(risk_appetite.lower(), self.risk_targets["moderate"])
        scores = expected_returns / returns.std().replace(0, np.nan)
        scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0.01)
        weights = scores / scores.sum()
        portfolio_vol = float(np.sqrt(np.dot(weights.T, np.dot(covariance, weights))))
        if portfolio_vol > target_volatility:
            weights = weights * (target_volatility / portfolio_vol)
            cash_weight = max(0.0, 1.0 - float(weights.sum()))
        else:
            cash_weight = 0.0
        weights = weights / max(float(weights.sum()), 1e-9) * (1 - cash_weight)
        allocation = []
        for ticker, weight in weights.items():
            latest_price = float(price_history[ticker]["Close"].iloc[-1])
            dollars = investment_amount * float(weight)
            allocation.append({"ticker": ticker, "weight": round(float(weight), 4), "amount": round(dollars, 2), "shares": round(dollars / latest_price, 4)})
        if cash_weight:
            allocation.append({"ticker": "CASH", "weight": round(cash_weight, 4), "amount": round(investment_amount * cash_weight, 2), "shares": 0})
        return {"risk_appetite": risk_appetite, "target_volatility": target_volatility, "expected_return": round(float(np.dot(weights, expected_returns)), 4), "expected_volatility": round(min(portfolio_vol, target_volatility), 4), "allocation": allocation}
