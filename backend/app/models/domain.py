from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base, TimestampMixin


class UserRole(StrEnum):
    user = "user"
    premium = "premium"
    admin = "admin"


class TransactionType(StrEnum):
    buy = "buy"
    sell = "sell"
    dividend = "dividend"
    deposit = "deposit"
    withdrawal = "withdrawal"


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    firebase_uid: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, name="user_role"), default=UserRole.user, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    profile: Mapped["Profile"] = relationship(back_populates="user", cascade="all, delete-orphan", uselist=False)
    watchlists: Mapped[list["Watchlist"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    portfolios: Mapped[list["Portfolio"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    notifications: Mapped[list["Notification"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Profile(Base, TimestampMixin):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(120))
    avatar_url: Mapped[str | None] = mapped_column(String(500))
    phone: Mapped[str | None] = mapped_column(String(40))
    country: Mapped[str | None] = mapped_column(String(80))
    timezone: Mapped[str | None] = mapped_column(String(80), default="UTC")
    risk_preference: Mapped[str | None] = mapped_column(String(32), default="balanced")

    user: Mapped[User] = relationship(back_populates="profile")


class Stock(Base, TimestampMixin):
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(24), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    exchange: Mapped[str | None] = mapped_column(String(64))
    sector: Mapped[str | None] = mapped_column(String(120))
    industry: Mapped[str | None] = mapped_column(String(120))
    currency: Mapped[str] = mapped_column(String(8), default="USD", nullable=False)
    country: Mapped[str | None] = mapped_column(String(80))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    watchlist_items: Mapped[list["WatchlistItem"]] = relationship(back_populates="stock", cascade="all, delete-orphan")
    holdings: Mapped[list["PortfolioHolding"]] = relationship(back_populates="stock", cascade="all, delete-orphan")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="stock")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="stock", cascade="all, delete-orphan")
    news: Mapped[list["MarketNews"]] = relationship(back_populates="stock", cascade="all, delete-orphan")
    sentiment: Mapped[list["SentimentAnalysis"]] = relationship(back_populates="stock", cascade="all, delete-orphan")
    risk_reports: Mapped[list["RiskReport"]] = relationship(back_populates="stock", cascade="all, delete-orphan")


class Watchlist(Base, TimestampMixin):
    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), default="Default", nullable=False)

    user: Mapped[User] = relationship(back_populates="watchlists")
    items: Mapped[list["WatchlistItem"]] = relationship(back_populates="watchlist", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("user_id", "name"),)


class WatchlistItem(Base, TimestampMixin):
    __tablename__ = "watchlist_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    watchlist_id: Mapped[int] = mapped_column(ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    watchlist: Mapped[Watchlist] = relationship(back_populates="items")
    stock: Mapped[Stock] = relationship(back_populates="watchlist_items")

    __table_args__ = (UniqueConstraint("watchlist_id", "stock_id"),)


class Portfolio(Base, TimestampMixin):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), default="Primary Portfolio", nullable=False)
    base_currency: Mapped[str] = mapped_column(String(8), default="USD", nullable=False)

    user: Mapped[User] = relationship(back_populates="portfolios")
    holdings: Mapped[list["PortfolioHolding"]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")


class PortfolioHolding(Base, TimestampMixin):
    __tablename__ = "portfolio_holdings"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    average_price: Mapped[float] = mapped_column(Float, nullable=False)

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")
    stock: Mapped[Stock] = relationship(back_populates="holdings")

    __table_args__ = (UniqueConstraint("portfolio_id", "stock_id"),)


class Transaction(Base, TimestampMixin):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    stock_id: Mapped[int | None] = mapped_column(ForeignKey("stocks.id", ondelete="SET NULL"))
    type: Mapped[TransactionType] = mapped_column(Enum(TransactionType, name="transaction_type"), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fees: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    portfolio: Mapped[Portfolio] = relationship(back_populates="transactions")
    stock: Mapped[Stock | None] = relationship(back_populates="transactions")


class PredictionModel(Base, TimestampMixin):
    __tablename__ = "prediction_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    version: Mapped[str] = mapped_column(String(40), default="1.0.0", nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model")


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("prediction_models.id", ondelete="SET NULL"))
    horizon_days: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    recommendation: Mapped[str] = mapped_column(String(16), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    user: Mapped[User | None] = relationship(back_populates="predictions")
    stock: Mapped[Stock] = relationship(back_populates="predictions")
    model: Mapped[PredictionModel | None] = relationship(back_populates="predictions")


class MarketNews(Base, TimestampMixin):
    __tablename__ = "market_news"

    id: Mapped[int] = mapped_column(primary_key=True)
    stock_id: Mapped[int | None] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    url: Mapped[str | None] = mapped_column(String(800))
    summary: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    stock: Mapped[Stock | None] = relationship(back_populates="news")
    sentiment: Mapped["SentimentAnalysis"] = relationship(back_populates="news", cascade="all, delete-orphan", uselist=False)


class SentimentAnalysis(Base, TimestampMixin):
    __tablename__ = "sentiment_analysis"

    id: Mapped[int] = mapped_column(primary_key=True)
    stock_id: Mapped[int | None] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"))
    news_id: Mapped[int | None] = mapped_column(ForeignKey("market_news.id", ondelete="CASCADE"), unique=True)
    sentiment: Mapped[str] = mapped_column(String(24), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), default="lexicon-v1", nullable=False)

    stock: Mapped[Stock | None] = relationship(back_populates="sentiment")
    news: Mapped[MarketNews | None] = relationship(back_populates="sentiment")


class RiskReport(Base, TimestampMixin):
    __tablename__ = "risk_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    stock_id: Mapped[int | None] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"))
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    volatility: Mapped[float] = mapped_column(Float, nullable=False)
    beta: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    value_at_risk: Mapped[float] = mapped_column(Float, nullable=False)
    report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    stock: Mapped[Stock | None] = relationship(back_populates="risk_reports")


class Alert(Base, TimestampMixin):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    stock_id: Mapped[int | None] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"))
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    operator: Mapped[str] = mapped_column(String(4), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    channel: Mapped[str] = mapped_column(String(32), default="email", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    user: Mapped[User] = relationship(back_populates="alerts")
    logs: Mapped[list["AlertLog"]] = relationship(back_populates="alert", cascade="all, delete-orphan")


class AlertLog(Base, TimestampMixin):
    __tablename__ = "alert_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False)
    triggered_value: Mapped[float] = mapped_column(Float, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    alert: Mapped[Alert] = relationship(back_populates="logs")


class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(180), default="New Copilot Chat", nullable=False)

    user: Mapped[User] = relationship(back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(120), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(120))
    resource_id: Mapped[str | None] = mapped_column(String(120))
    ip_address: Mapped[str | None] = mapped_column(String(80))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Notification(Base, TimestampMixin):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    channel: Mapped[str] = mapped_column(String(32), default="in_app", nullable=False)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped[User] = relationship(back_populates="notifications")


Index("ix_transactions_portfolio_executed", Transaction.portfolio_id, Transaction.executed_at)
Index("ix_predictions_stock_created", Prediction.stock_id, Prediction.created_at)
Index("ix_market_news_stock_published", MarketNews.stock_id, MarketNews.published_at)
