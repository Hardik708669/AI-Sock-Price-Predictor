from app.services.market_data import MarketDataService
from app.services.prediction import PredictionService
from app.services.sentiment import SentimentService


class AssistantService:
    def __init__(self, market: MarketDataService, prediction: PredictionService, sentiment: SentimentService):
        self.market = market
        self.prediction = prediction
        self.sentiment = sentiment

    def answer(self, message: str, symbols: list[str]) -> dict:
        focus = symbols[0] if symbols else self._extract_symbol(message)
        used_data = ["market overview", "technical indicators", "multi-model prediction", "news sentiment"]
        overview = self.market.overview(focus)
        prediction = self.prediction.predict(focus)
        sentiment = self.sentiment.aggregate(focus)
        answer = (
            f"For {overview.symbol}, StockVision AI currently rates the setup as {prediction.recommendation}. "
            f"The live price is {overview.live_price:.2f}, market sentiment is {sentiment['sentiment']}, "
            f"and the AI model set shows confidence near "
            f"{sum(model.confidence for model in prediction.models) / len(prediction.models):.1f}%. "
            f"{prediction.explanation} Risk should be managed with position sizing and alerts."
        )
        return {"answer": answer, "used_data": used_data}

    def _extract_symbol(self, message: str) -> str:
        for token in message.replace("?", " ").replace(",", " ").split():
            if token.isupper() and 1 <= len(token) <= 12:
                return token
        return "AAPL"
