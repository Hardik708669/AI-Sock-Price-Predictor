from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.domain import User
from app.schemas import StockOverview, StockSearchResult
from app.services.market_data import MarketDataService, get_market_data_service

router = APIRouter()


@router.get("/search", response_model=list[StockSearchResult])
def search_stocks(q: str, market_data: MarketDataService = Depends(get_market_data_service), _user: User = Depends(get_current_user)) -> list[dict]:
    return market_data.search(q)


@router.get("/history")
def stock_history(ticker: str, period: str = "1y", market_data: MarketDataService = Depends(get_market_data_service), _user: User = Depends(get_current_user)) -> dict:
    return {"ticker": ticker.upper(), "candles": market_data.candles(ticker, period)}


@router.get("/live")
def stock_live(ticker: str, market_data: MarketDataService = Depends(get_market_data_service), _user: User = Depends(get_current_user)) -> dict:
    overview = market_data.overview(ticker)
    return {"ticker": overview.symbol, "price": overview.live_price, "currency": overview.currency}


@router.get("/{ticker}", response_model=StockOverview)
def stock_detail(ticker: str, market_data: MarketDataService = Depends(get_market_data_service), _user: User = Depends(get_current_user)) -> StockOverview:
    try:
        return market_data.overview(ticker)
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/{ticker}/candles")
def stock_candles(ticker: str, period: str = "1y", market_data: MarketDataService = Depends(get_market_data_service), _user: User = Depends(get_current_user)) -> dict:
    return {"ticker": ticker.upper(), "candles": market_data.candles(ticker, period)}
