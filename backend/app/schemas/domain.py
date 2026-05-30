from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

from app.schemas.common import ORMModel, Role


class ProfileRead(ORMModel):
    full_name: str | None = None
    avatar_url: str | None = None
    phone: str | None = None
    country: str | None = None
    timezone: str | None = None
    risk_preference: str | None = None


class ProfileUpdate(BaseModel):
    full_name: str | None = Field(default=None, max_length=120)
    avatar_url: str | None = Field(default=None, max_length=500)
    phone: str | None = Field(default=None, max_length=40)
    country: str | None = Field(default=None, max_length=80)
    timezone: str | None = Field(default=None, max_length=80)
    risk_preference: Literal["conservative", "balanced", "aggressive"] | None = None


class UserRead(ORMModel):
    id: int
    email: EmailStr
    role: Role
    is_active: bool
    is_verified: bool
    profile: ProfileRead | None = None


class StockSearchResult(BaseModel):
    symbol: str
    name: str
    exchange: str
    sector: str | None = None


class StockRead(ORMModel):
    id: int
    ticker: str
    name: str
    exchange: str | None = None
    sector: str | None = None
    currency: str = "USD"


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


class PortfolioHoldingIn(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    quantity: float = Field(gt=0)
    average_price: float = Field(gt=0)


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


class PortfolioRead(ORMModel):
    id: int
    name: str
    base_currency: str
    created_at: datetime


class TransactionCreate(BaseModel):
    ticker: str
    type: Literal["buy", "sell", "dividend", "deposit", "withdrawal"]
    quantity: float = Field(gt=0)
    price: float = Field(ge=0)
    fees: float = Field(default=0, ge=0)


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


class PredictRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    horizon_days: int = Field(default=30, ge=1, le=365)


class SentimentItem(BaseModel):
    title: str
    source: str
    sentiment: Literal["Bullish", "Bearish", "Neutral"]
    score: float
    summary: str
    published_at: datetime


class AlertRequest(BaseModel):
    symbol: str
    metric: str
    operator: Literal[">", "<", ">=", "<="]
    threshold: float
    channel: Literal["email", "in_app", "sms", "webhook"] = "email"


class AlertRead(ORMModel):
    id: int
    metric: str
    operator: str
    threshold: float
    channel: str
    is_active: bool


class AssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    symbols: list[str] = Field(default_factory=list, max_length=20)


class AssistantResponse(BaseModel):
    answer: str
    used_data: list[str]


class AdminUserUpdate(BaseModel):
    role: Role | None = None
    is_active: bool | None = None


class AIIntelligenceRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)


class PortfolioOptimizeRequest(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=30)
    investment_amount: float = Field(gt=0)
    risk_appetite: Literal["conservative", "moderate", "aggressive"] = "moderate"


class RetrainRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    period: str = "5y"
