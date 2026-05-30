from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_sentiment_service
from app.models.domain import User
from app.services.sentiment import SentimentService

router = APIRouter()


@router.get("")
def get_news(ticker: str | None = None, _user: User = Depends(get_current_user), sentiment: SentimentService = Depends(get_sentiment_service)) -> dict:
    symbol = ticker or "MARKET"
    return {"items": [item.model_dump() for item in sentiment.items(symbol)]}


@router.get("/sentiment")
def get_news_sentiment(ticker: str = "AAPL", _user: User = Depends(get_current_user), sentiment: SentimentService = Depends(get_sentiment_service)) -> dict:
    return sentiment.aggregate(ticker)
