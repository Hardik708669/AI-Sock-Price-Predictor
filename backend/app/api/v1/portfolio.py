from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database import get_db
from app.models.domain import User
from app.repositories import ActivityRepository, PortfolioRepository, StockRepository, TransactionRepository
from app.schemas import PortfolioHoldingIn
from app.services.market_data import get_market_data_service

router = APIRouter()


@router.get("")
def get_portfolio(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    portfolio = PortfolioRepository(db).primary_for_user(current_user.id)
    market = get_market_data_service()
    holdings = []
    total = 0.0
    for holding in portfolio.holdings:
        try:
            price = market.overview(holding.stock.ticker).live_price
        except Exception:
            price = holding.average_price
        value = holding.quantity * price
        total += value
        holdings.append({"ticker": holding.stock.ticker, "quantity": holding.quantity, "average_price": holding.average_price, "current_price": price, "value": value})
    return {"portfolio": {"id": portfolio.id, "name": portfolio.name, "base_currency": portfolio.base_currency}, "total_value": total, "holdings": holdings}


@router.post("/add")
def add_holding(payload: PortfolioHoldingIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    stocks = StockRepository(db)
    portfolio = PortfolioRepository(db).primary_for_user(current_user.id)
    stock = stocks.get_or_create(payload.ticker)
    holding = PortfolioRepository(db).add_holding(portfolio, stock, payload.quantity, payload.average_price)
    ActivityRepository(db).audit("portfolio.add", current_user.id, "stock", stock.ticker)
    db.commit()
    return {"message": "Holding added", "holding_id": holding.id}


@router.post("/remove")
def remove_holding(ticker: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    stock = StockRepository(db).get_by_ticker(ticker)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
    removed = PortfolioRepository(db).remove_holding(PortfolioRepository(db).primary_for_user(current_user.id), stock)
    if not removed:
        raise HTTPException(status_code=404, detail="Holding not found")
    ActivityRepository(db).audit("portfolio.remove", current_user.id, "stock", stock.ticker)
    db.commit()
    return {"message": "Holding removed"}


@router.get("/transactions")
def transactions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    rows = TransactionRepository(db).list_for_user(current_user.id)
    return {"items": [{"id": row.id, "ticker": row.stock.ticker if row.stock else None, "type": row.type.value, "quantity": row.quantity, "price": row.price, "executed_at": row.executed_at} for row in rows]}
