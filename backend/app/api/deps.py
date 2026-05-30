from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.database import get_db
from app.models.domain import User, UserRole
from app.repositories import UserRepository
from app.services.market_data import MarketDataService, get_market_data_service
from app.services.prediction import PredictionService
from app.services.risk import RiskService
from app.services.sentiment import SentimentService

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from error
    user = UserRepository(db).get_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive or missing user")
    return user


def require_roles(*roles: UserRole):
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user

    return dependency


def require_admin(current_user: User = Depends(require_roles(UserRole.admin))) -> User:
    return current_user


def get_prediction_service(
    market_data: MarketDataService = Depends(get_market_data_service),
) -> PredictionService:
    return PredictionService(market_data)


def get_risk_service(market_data: MarketDataService = Depends(get_market_data_service)) -> RiskService:
    return RiskService(market_data)


def get_sentiment_service() -> SentimentService:
    return SentimentService()
