class InvestmentCopilot:
    def recommend(self, ticker: str, prediction: dict, technicals: dict, sentiment: dict, risk: dict, trend: dict, anomalies: dict) -> dict:
        score = 0
        reasons: list[str] = []
        if prediction["recommendation"] == "BUY":
            score += 2
            reasons.append("model ensemble expects upside")
        elif prediction["recommendation"] == "SELL":
            score -= 2
            reasons.append("model ensemble expects downside")
        if technicals["signal"] == "BUY":
            score += 1
            reasons.append("technical setup is constructive")
        elif technicals["signal"] == "SELL":
            score -= 1
            reasons.append("technical setup is weak")
        if sentiment["sentiment"] == "Bullish":
            score += 1
            reasons.append("news sentiment is bullish")
        elif sentiment["sentiment"] == "Bearish":
            score -= 1
            reasons.append("news sentiment is bearish")
        if risk["risk_score"] > 70:
            score -= 1
            reasons.append("risk score is elevated")
        if trend["regime"] == "Bull Market":
            score += 1
            reasons.append("market regime is bullish")
        elif trend["regime"] == "Bear Market":
            score -= 1
            reasons.append("market regime is bearish")
        if anomalies["latest_anomaly"]:
            score -= 1
            reasons.append("latest market behavior is anomalous")
        action = "BUY" if score >= 3 else "SELL" if score <= -2 else "HOLD"
        narrative = (
            f"For {ticker.upper()}, StockVision AI recommends {action}. "
            f"The ensemble target is {prediction['ensemble_prediction']} versus the current price {prediction['current_price']}. "
            f"Key reasons: {', '.join(reasons) if reasons else 'signals are balanced'}. "
            f"Risk is {risk['risk_level']} with volatility {risk['volatility']} and maximum drawdown {risk['maximum_drawdown']}."
        )
        return {"ticker": ticker.upper(), "recommendation": action, "score": score, "reasons": reasons, "analysis": narrative}
