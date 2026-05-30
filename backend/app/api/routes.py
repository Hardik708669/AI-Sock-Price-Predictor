from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_prediction_service, get_risk_service, get_sentiment_service
from app.core.firebase import verify_firebase_token
from app.core.security import create_access_token, hash_password
from app.schemas import (
    AlertRequest,
    AssistantRequest,
    AssistantResponse,
    FirebaseLoginRequest,
    LoginRequest,
    PortfolioRequest,
    PortfolioResponse,
    RegisterRequest,
    StockOverview,
    StockSearchResult,
    TokenResponse,
)
from app.services.assistant import AssistantService
from app.services.market_data import MarketDataService, get_market_data_service
from app.services.portfolio import PortfolioService
from app.services.prediction import PredictionService
from app.services.risk import RiskService
from app.services.sentiment import SentimentService

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "StockVision AI"}


@router.post("/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest) -> TokenResponse:
    password_hash = hash_password(payload.password)
    token = create_access_token(payload.email, {"name": payload.name, "hash": password_hash[:12]})
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    token = create_access_token(payload.email)
    return TokenResponse(access_token=token)


@router.post("/auth/firebase", response_model=TokenResponse)
def firebase_login(payload: FirebaseLoginRequest) -> TokenResponse:
    try:
        decoded = verify_firebase_token(payload.id_token)
    except Exception as error:
        raise HTTPException(status_code=401, detail="Invalid Firebase token") from error
    token = create_access_token(decoded["uid"], {"email": decoded.get("email")})
    return TokenResponse(access_token=token)


@router.get("/stocks/search", response_model=list[StockSearchResult])
def search_stocks(
    q: str,
    market_data: MarketDataService = Depends(get_market_data_service),
) -> list[dict]:
    return market_data.search(q)


@router.get("/stocks/{symbol}/overview", response_model=StockOverview)
def stock_overview(
    symbol: str,
    market_data: MarketDataService = Depends(get_market_data_service),
) -> StockOverview:
    try:
        return market_data.overview(symbol)
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/stocks/{symbol}/candles")
def stock_candles(
    symbol: str,
    period: str = "1y",
    market_data: MarketDataService = Depends(get_market_data_service),
) -> dict:
    try:
        return {"symbol": symbol.upper(), "candles": market_data.candles(symbol, period)}
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/stocks/{symbol}/technicals")
def stock_technicals(
    symbol: str,
    market_data: MarketDataService = Depends(get_market_data_service),
) -> dict:
    try:
        return market_data.technicals(symbol).model_dump()
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/predictions/{symbol}")
def predictions(
    symbol: str,
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> dict:
    try:
        return prediction_service.predict(symbol).model_dump()
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/sentiment/{symbol}")
def sentiment(
    symbol: str,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> dict:
    return sentiment_service.aggregate(symbol)


@router.get("/risk/{symbol}")
def risk(
    symbol: str,
    risk_service: RiskService = Depends(get_risk_service),
) -> dict:
    try:
        return risk_service.analyze(symbol)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/portfolio/analyze", response_model=PortfolioResponse)
def analyze_portfolio(payload: PortfolioRequest) -> PortfolioResponse:
    return PortfolioService().analyze(payload)


@router.get("/watchlist")
def watchlist() -> dict:
    return {
        "items": [
            {"symbol": "AAPL", "price": 212.44, "daily_change": 1.24, "ai_rating": "BUY"},
            {"symbol": "NVDA", "price": 139.89, "daily_change": -0.42, "ai_rating": "HOLD"},
            {"symbol": "RELIANCE.NS", "price": 2910.3, "daily_change": 0.76, "ai_rating": "BUY"},
        ]
    }


@router.post("/alerts")
def create_alert(payload: AlertRequest) -> dict:
    return {
        "id": "alert-demo-001",
        "message": f"Alert created: {payload.symbol} {payload.metric} {payload.operator} {payload.threshold}",
        "channel": payload.channel,
        "active": True,
    }


@router.get("/heatmap")
def market_heatmap() -> dict:
    return {
        "categories": [
            {"sector": "Technology", "change": 1.8, "symbols": ["AAPL", "MSFT", "NVDA"]},
            {"sector": "Banking", "change": 0.7, "symbols": ["JPM", "HDFCBANK.NS"]},
            {"sector": "Healthcare", "change": -0.4, "symbols": ["JNJ", "PFE"]},
            {"sector": "Energy", "change": 1.1, "symbols": ["XOM", "RELIANCE.NS"]},
            {"sector": "Consumer", "change": -0.2, "symbols": ["TSLA", "AMZN"]},
        ]
    }


@router.get("/dashboard")
def dashboard() -> dict:
    return {
        "portfolio_value": 128940.55,
        "daily_profit_loss": 1842.2,
        "ai_confidence_score": 84,
        "market_sentiment": "Bullish",
        "prediction_center": [
            {"symbol": "AAPL", "signal": "BUY", "confidence": 86},
            {"symbol": "TSLA", "signal": "HOLD", "confidence": 72},
            {"symbol": "NVDA", "signal": "BUY", "confidence": 89},
        ],
    }


@router.post("/assistant", response_model=AssistantResponse)
def assistant(
    payload: AssistantRequest,
    market_data: MarketDataService = Depends(get_market_data_service),
    prediction_service: PredictionService = Depends(get_prediction_service),
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> dict:
    service = AssistantService(market_data, prediction_service, sentiment_service)
    return service.answer(payload.message, payload.symbols)
