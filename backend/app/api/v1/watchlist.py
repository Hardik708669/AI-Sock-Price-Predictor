from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database import get_db
from app.models.domain import User
from app.repositories import ActivityRepository, StockRepository, WatchlistRepository

router = APIRouter()


@router.get("")
def get_watchlist(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    watchlist = WatchlistRepository(db).default_for_user(current_user.id)
    return {"id": watchlist.id, "name": watchlist.name, "items": [{"ticker": item.stock.ticker, "name": item.stock.name, "sector": item.stock.sector} for item in watchlist.items]}


@router.post("/add", status_code=status.HTTP_201_CREATED)
def add_watchlist_item(ticker: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    stock = StockRepository(db).get_or_create(ticker)
    watchlist = WatchlistRepository(db).default_for_user(current_user.id)
    item = WatchlistRepository(db).add(watchlist, stock)
    ActivityRepository(db).audit("watchlist.add", current_user.id, "stock", stock.ticker)
    db.commit()
    return {"message": "Stock added to watchlist", "item_id": item.id}


@router.delete("/remove")
def remove_watchlist_item(ticker: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    stock = StockRepository(db).get_by_ticker(ticker)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
    removed = WatchlistRepository(db).remove(WatchlistRepository(db).default_for_user(current_user.id), stock)
    if not removed:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    ActivityRepository(db).audit("watchlist.remove", current_user.id, "stock", stock.ticker)
    db.commit()
    return {"message": "Stock removed from watchlist"}
