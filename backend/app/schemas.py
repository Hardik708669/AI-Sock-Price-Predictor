from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class RegisterRequest(LoginRequest):
    name: str = Field(min_length=2)


class FirebaseLoginRequest(BaseModel):
    id_token: str


class StockSearchResult(BaseModel):
    symbol: str
    name: str
    exchange: str
    sector: str | None = None


class StockOverview(BaseModel):
    symbol: str
    name: str
    live_price: float
    currency: str
    market_cap: float | None
    volume: float | None
    pe_ratio: float | None
    eps: float | None
    dividend_yield: float | None
    sector: str
    summary: str


class Candle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TechnicalSignals(BaseModel):
    rsi: float
    macd: float
    ema_20: float
    sma_50: float
    bollinger_upper: float
    bollinger_lower: float
    vwap: float


class ModelPrediction(BaseModel):
    model_name: str
    predicted_price: float
    confidence: float
    accuracy: float


class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    models: list[ModelPrediction]
    recommendation: Literal["BUY", "SELL", "HOLD"]
    explanation: str
    feature_importance: list[dict[str, float | str]]


class SentimentItem(BaseModel):
    title: str
    source: str
    sentiment: Literal["Bullish", "Bearish", "Neutral"]
    score: float
    summary: str
    published_at: datetime


class PortfolioHolding(BaseModel):
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    sector: str


class PortfolioRequest(BaseModel):
    holdings: list[PortfolioHolding]


class PortfolioResponse(BaseModel):
    total_investment: float
    current_value: float
    profit_loss: float
    allocation: list[dict[str, float | str]]
    portfolio_health_score: float
    risk_score: float
    rebalance_actions: list[str]


class AlertRequest(BaseModel):
    symbol: str
    metric: str
    operator: Literal[">", "<", ">=", "<="]
    threshold: float
    channel: Literal["email", "in_app"] = "email"


class AssistantRequest(BaseModel):
    message: str
    symbols: list[str] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    answer: str
    used_data: list[str]
