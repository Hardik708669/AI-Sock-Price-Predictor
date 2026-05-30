from collections import defaultdict

from app.schemas import PortfolioRequest, PortfolioResponse


class PortfolioService:
    def analyze(self, request: PortfolioRequest) -> PortfolioResponse:
        total_investment = sum(item.quantity * item.average_price for item in request.holdings)
        current_value = sum(item.quantity * item.current_price for item in request.holdings)
        profit_loss = current_value - total_investment
        sector_values: dict[str, float] = defaultdict(float)
        symbol_values: dict[str, float] = {}
        for holding in request.holdings:
            value = holding.quantity * holding.current_price
            sector_values[holding.sector] += value
            symbol_values[holding.symbol] = value

        allocation = [
            {"name": sector, "value": round(value / current_value * 100, 2) if current_value else 0}
            for sector, value in sector_values.items()
        ]
        concentration = max((value / current_value for value in symbol_values.values()), default=0)
        risk_score = min(100, 35 + concentration * 45 + max(0, -profit_loss / max(total_investment, 1)) * 20)
        health_score = max(0, min(100, 100 - risk_score + min(12, max(0, profit_loss / max(total_investment, 1) * 100))))
        rebalance_actions = []
        for symbol, value in symbol_values.items():
            weight = value / current_value if current_value else 0
            if weight > 0.35:
                rebalance_actions.append(f"Trim {symbol} to reduce concentration below 35%.")
            elif weight < 0.08 and len(symbol_values) > 4:
                rebalance_actions.append(f"Review {symbol}; position may be too small to impact returns.")
        if not rebalance_actions:
            rebalance_actions.append("Portfolio is balanced. Rebalance only if target allocation drifts above 5%.")
        return PortfolioResponse(
            total_investment=round(total_investment, 2),
            current_value=round(current_value, 2),
            profit_loss=round(profit_loss, 2),
            allocation=allocation,
            portfolio_health_score=round(health_score, 2),
            risk_score=round(risk_score, 2),
            rebalance_actions=rebalance_actions,
        )
