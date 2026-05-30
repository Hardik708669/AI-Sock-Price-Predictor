from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models.domain import (
    Alert,
    AuditLog,
    ChatMessage,
    ChatSession,
    Portfolio,
    PortfolioHolding,
    Prediction,
    Profile,
    Stock,
    Transaction,
    User,
    Watchlist,
    WatchlistItem,
)


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: int) -> User | None:
        return self.db.scalar(select(User).options(selectinload(User.profile)).where(User.id == user_id))

    def get_by_email(self, email: str) -> User | None:
        return self.db.scalar(select(User).options(selectinload(User.profile)).where(func.lower(User.email) == email.lower()))

    def get_by_firebase_uid(self, uid: str) -> User | None:
        return self.db.scalar(select(User).where(User.firebase_uid == uid))

    def create(self, email: str, password_hash: str | None, full_name: str | None, firebase_uid: str | None = None) -> User:
        user = User(email=email.lower(), password_hash=password_hash, firebase_uid=firebase_uid, is_verified=bool(firebase_uid))
        user.profile = Profile(full_name=full_name)
        user.watchlists.append(Watchlist(name="Default"))
        user.portfolios.append(Portfolio(name="Primary Portfolio"))
        self.db.add(user)
        self.db.flush()
        return user

    def list(self, limit: int = 50, offset: int = 0) -> list[User]:
        return list(self.db.scalars(select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)))


class StockRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_ticker(self, ticker: str) -> Stock | None:
        return self.db.scalar(select(Stock).where(Stock.ticker == ticker.upper()))

    def get_or_create(self, ticker: str, name: str | None = None, exchange: str | None = None, sector: str | None = None) -> Stock:
        stock = self.get_by_ticker(ticker)
        if stock:
            return stock
        stock = Stock(ticker=ticker.upper(), name=name or ticker.upper(), exchange=exchange, sector=sector)
        self.db.add(stock)
        self.db.flush()
        return stock

    def search(self, query: str, limit: int = 20) -> list[Stock]:
        like = f"%{query.upper()}%"
        return list(self.db.scalars(select(Stock).where((Stock.ticker.ilike(like)) | (Stock.name.ilike(like))).limit(limit)))


class PortfolioRepository:
    def __init__(self, db: Session):
        self.db = db

    def primary_for_user(self, user_id: int) -> Portfolio:
        portfolio = self.db.scalar(select(Portfolio).options(selectinload(Portfolio.holdings).selectinload(PortfolioHolding.stock)).where(Portfolio.user_id == user_id).order_by(Portfolio.id))
        if portfolio:
            return portfolio
        portfolio = Portfolio(user_id=user_id, name="Primary Portfolio")
        self.db.add(portfolio)
        self.db.flush()
        return portfolio

    def add_holding(self, portfolio: Portfolio, stock: Stock, quantity: float, average_price: float) -> PortfolioHolding:
        existing = self.db.scalar(select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio.id, PortfolioHolding.stock_id == stock.id))
        if existing:
            total_quantity = existing.quantity + quantity
            existing.average_price = ((existing.quantity * existing.average_price) + (quantity * average_price)) / total_quantity
            existing.quantity = total_quantity
            return existing
        holding = PortfolioHolding(portfolio_id=portfolio.id, stock_id=stock.id, quantity=quantity, average_price=average_price)
        self.db.add(holding)
        self.db.flush()
        return holding

    def remove_holding(self, portfolio: Portfolio, stock: Stock) -> bool:
        holding = self.db.scalar(select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio.id, PortfolioHolding.stock_id == stock.id))
        if not holding:
            return False
        self.db.delete(holding)
        return True


class WatchlistRepository:
    def __init__(self, db: Session):
        self.db = db

    def default_for_user(self, user_id: int) -> Watchlist:
        watchlist = self.db.scalar(select(Watchlist).options(selectinload(Watchlist.items).selectinload(WatchlistItem.stock)).where(Watchlist.user_id == user_id).order_by(Watchlist.id))
        if watchlist:
            return watchlist
        watchlist = Watchlist(user_id=user_id, name="Default")
        self.db.add(watchlist)
        self.db.flush()
        return watchlist

    def add(self, watchlist: Watchlist, stock: Stock) -> WatchlistItem:
        item = self.db.scalar(select(WatchlistItem).where(WatchlistItem.watchlist_id == watchlist.id, WatchlistItem.stock_id == stock.id))
        if item:
            return item
        item = WatchlistItem(watchlist_id=watchlist.id, stock_id=stock.id)
        self.db.add(item)
        self.db.flush()
        return item

    def remove(self, watchlist: Watchlist, stock: Stock) -> bool:
        item = self.db.scalar(select(WatchlistItem).where(WatchlistItem.watchlist_id == watchlist.id, WatchlistItem.stock_id == stock.id))
        if not item:
            return False
        self.db.delete(item)
        return True


class ActivityRepository:
    def __init__(self, db: Session):
        self.db = db

    def audit(self, action: str, actor_user_id: int | None = None, resource_type: str | None = None, resource_id: str | None = None, metadata: dict[str, Any] | None = None, ip_address: str | None = None, user_agent: str | None = None) -> AuditLog:
        log = AuditLog(
            action=action,
            actor_user_id=actor_user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata_json=metadata or {},
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(log)
        self.db.flush()
        return log


class IntelligenceRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_prediction(self, user_id: int | None, stock: Stock, predicted_price: float, confidence: float, recommendation: str, explanation: str, features: dict[str, Any], horizon_days: int) -> Prediction:
        prediction = Prediction(
            user_id=user_id,
            stock_id=stock.id,
            predicted_price=predicted_price,
            confidence=confidence,
            recommendation=recommendation,
            explanation=explanation,
            features=features,
            horizon_days=horizon_days,
        )
        self.db.add(prediction)
        self.db.flush()
        return prediction

    def prediction_history(self, user_id: int, limit: int = 50) -> list[Prediction]:
        return list(self.db.scalars(select(Prediction).where(Prediction.user_id == user_id).order_by(Prediction.created_at.desc()).limit(limit)))


class AlertRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, user_id: int, stock: Stock | None, metric: str, operator: str, threshold: float, channel: str) -> Alert:
        alert = Alert(user_id=user_id, stock_id=stock.id if stock else None, metric=metric, operator=operator, threshold=threshold, channel=channel)
        self.db.add(alert)
        self.db.flush()
        return alert

    def list_for_user(self, user_id: int) -> list[Alert]:
        return list(self.db.scalars(select(Alert).where(Alert.user_id == user_id).order_by(Alert.created_at.desc())))


class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_session(self, user_id: int) -> ChatSession:
        session = self.db.scalar(select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.updated_at.desc()))
        if session:
            return session
        session = ChatSession(user_id=user_id)
        self.db.add(session)
        self.db.flush()
        return session

    def add_message(self, session: ChatSession, role: str, content: str, context: dict[str, Any] | None = None) -> ChatMessage:
        message = ChatMessage(session_id=session.id, role=role, content=content, context=context or {})
        self.db.add(message)
        self.db.flush()
        return message


class TransactionRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_for_user(self, user_id: int) -> list[Transaction]:
        return list(
            self.db.scalars(
                select(Transaction)
                .join(Portfolio)
                .where(Portfolio.user_id == user_id)
                .order_by(Transaction.executed_at.desc())
                .options(selectinload(Transaction.stock))
            )
        )
