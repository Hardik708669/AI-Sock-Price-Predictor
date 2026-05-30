from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_prediction_service, get_sentiment_service
from app.database import get_db
from app.models.domain import User
from app.repositories import ActivityRepository, ChatRepository
from app.schemas import AssistantRequest, AssistantResponse
from app.services.assistant import AssistantService
from app.services.market_data import MarketDataService, get_market_data_service
from app.services.prediction import PredictionService
from app.services.sentiment import SentimentService

router = APIRouter()


@router.post("/chat", response_model=AssistantResponse)
def chat(
    payload: AssistantRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    market_data: MarketDataService = Depends(get_market_data_service),
    prediction_service: PredictionService = Depends(get_prediction_service),
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> dict:
    service = AssistantService(market_data, prediction_service, sentiment_service)
    response = service.answer(payload.message, payload.symbols)
    chats = ChatRepository(db)
    session = chats.get_or_create_session(current_user.id)
    chats.add_message(session, "user", payload.message, {"symbols": payload.symbols})
    chats.add_message(session, "assistant", response["answer"], {"used_data": response["used_data"]})
    ActivityRepository(db).audit("copilot.chat", current_user.id, "chat_session", str(session.id))
    db.commit()
    return response


@router.post("/assistant", response_model=AssistantResponse)
def assistant_alias(payload: AssistantRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db), market_data: MarketDataService = Depends(get_market_data_service), prediction_service: PredictionService = Depends(get_prediction_service), sentiment_service: SentimentService = Depends(get_sentiment_service)) -> dict:
    return chat(payload, current_user, db, market_data, prediction_service, sentiment_service)
