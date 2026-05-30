from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_prediction_service
from app.database import get_db
from app.models.domain import User
from app.repositories import ActivityRepository, IntelligenceRepository, StockRepository
from app.schemas import PredictRequest
from app.services.prediction import PredictionService

router = APIRouter()


@router.post("/predict")
def predict(payload: PredictRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db), prediction_service: PredictionService = Depends(get_prediction_service)) -> dict:
    try:
        result = prediction_service.predict(payload.ticker)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    stock = StockRepository(db).get_or_create(payload.ticker)
    average_price = sum(model.predicted_price for model in result.models) / len(result.models)
    confidence = sum(model.confidence for model in result.models) / len(result.models)
    IntelligenceRepository(db).save_prediction(
        current_user.id,
        stock,
        average_price,
        confidence,
        result.recommendation,
        result.explanation,
        {"feature_importance": result.feature_importance, "models": [model.model_dump() for model in result.models]},
        payload.horizon_days,
    )
    ActivityRepository(db).audit("prediction.create", current_user.id, "stock", stock.ticker)
    db.commit()
    return result.model_dump()


@router.get("/prediction/history")
def prediction_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    history = IntelligenceRepository(db).prediction_history(current_user.id)
    return {
        "items": [
            {
                "id": item.id,
                "ticker": item.stock.ticker,
                "predicted_price": item.predicted_price,
                "confidence": item.confidence,
                "recommendation": item.recommendation,
                "created_at": item.created_at,
            }
            for item in history
        ]
    }


@router.get("/predictions/{ticker}")
def legacy_prediction(ticker: str, _user: User = Depends(get_current_user), prediction_service: PredictionService = Depends(get_prediction_service)) -> dict:
    return prediction_service.predict(ticker).model_dump()
