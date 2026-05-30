from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user, get_risk_service
from app.models.domain import User
from app.services.risk import RiskService

router = APIRouter()


@router.get("/risk-analysis")
def risk_analysis(ticker: str = "AAPL", _user: User = Depends(get_current_user), risk_service: RiskService = Depends(get_risk_service)) -> dict:
    try:
        return risk_service.analyze(ticker)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/risk/{ticker}")
def legacy_risk(ticker: str, _user: User = Depends(get_current_user), risk_service: RiskService = Depends(get_risk_service)) -> dict:
    return risk_service.analyze(ticker)
