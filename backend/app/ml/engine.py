from functools import lru_cache

from app.ml.anomaly_detection import AnomalyDetectionEngine
from app.ml.copilot import InvestmentCopilot
from app.ml.data_pipeline import MarketDataPipeline
from app.ml.explainability import ExplainableAIEngine
from app.ml.forecasting import ForecastingEngine
from app.ml.optimizer import PortfolioOptimizer
from app.ml.prediction_engine import StockPredictionEngine
from app.ml.risk_engine import RiskAnalysisEngine
from app.ml.sentiment_engine import FinancialSentimentEngine
from app.ml.technical_analysis import TechnicalAnalysisEngine
from app.ml.trend_detection import MarketTrendDetector


class StockVisionAIEngine:
    def __init__(self) -> None:
        self.data = MarketDataPipeline()
        self.predictions = StockPredictionEngine()
        self.technicals = TechnicalAnalysisEngine()
        self.sentiment = FinancialSentimentEngine()
        self.trends = MarketTrendDetector()
        self.risk = RiskAnalysisEngine()
        self.optimizer = PortfolioOptimizer()
        self.explainability = ExplainableAIEngine()
        self.anomalies = AnomalyDetectionEngine()
        self.forecasting = ForecastingEngine()
        self.copilot = InvestmentCopilot()

    def intelligence_report(self, ticker: str) -> dict:
        data = self.data.historical_prices(ticker)
        benchmark = self.data.historical_prices("^GSPC", period="5y")
        prediction = self.predictions.predict(data)
        best_model_name = max(prediction["models"], key=lambda item: item["confidence"])["model"]
        model = prediction["trained_models"].get(best_model_name) or next(iter(prediction["trained_models"].values()))
        explanation = self.explainability.explain(model, prediction["feature_frame"])
        technicals = self.technicals.signal(data)
        sentiment = self.sentiment.analyze_news(self.data.financial_news(ticker))
        trend = self.trends.detect(data)
        risk = self.risk.analyze(data, benchmark)
        anomalies = self.anomalies.detect(data)
        forecasts = self.forecasting.forecast(data)
        recommendation = self.copilot.recommend(ticker, prediction, technicals, sentiment, risk, trend, anomalies)
        return {
            "ticker": ticker.upper(),
            "prediction": {key: value for key, value in prediction.items() if key not in {"trained_models", "feature_frame", "target"}},
            "technical_analysis": technicals,
            "sentiment": sentiment,
            "trend": trend,
            "risk": risk,
            "anomalies": anomalies,
            "forecasts": forecasts,
            "explainable_ai": explanation,
            "copilot": recommendation,
        }

    def optimize_portfolio(self, tickers: list[str], investment_amount: float, risk_appetite: str) -> dict:
        history = {ticker.upper(): self.data.historical_prices(ticker) for ticker in tickers}
        return self.optimizer.optimize(history, investment_amount, risk_appetite)


@lru_cache
def get_ai_engine() -> StockVisionAIEngine:
    return StockVisionAIEngine()
