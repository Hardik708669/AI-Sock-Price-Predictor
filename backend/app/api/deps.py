from fastapi import Depends

from app.services.market_data import MarketDataService, get_market_data_service
from app.services.prediction import PredictionService
from app.services.risk import RiskService
from app.services.sentiment import SentimentService


def get_prediction_service(
    market_data: MarketDataService = Depends(get_market_data_service),
) -> PredictionService:
    return PredictionService(market_data)


def get_risk_service(market_data: MarketDataService = Depends(get_market_data_service)) -> RiskService:
    return RiskService(market_data)


def get_sentiment_service() -> SentimentService:
    return SentimentService()
