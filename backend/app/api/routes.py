from fastapi import APIRouter

from app.api.v1 import admin, ai_engine, alerts, auth, copilot, news, portfolio, predictions, risk, stocks, users, watchlist

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(users.router, prefix="/users", tags=["Users"])
router.include_router(stocks.router, prefix="/stocks", tags=["Stocks"])
router.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])
router.include_router(watchlist.router, prefix="/watchlist", tags=["Watchlist"])
router.include_router(predictions.router, tags=["Predictions"])
router.include_router(news.router, prefix="/news", tags=["News"])
router.include_router(risk.router, tags=["Risk"])
router.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
router.include_router(copilot.router, prefix="/copilot", tags=["AI Copilot"])
router.include_router(admin.router, prefix="/admin", tags=["Admin"])
router.include_router(ai_engine.router, prefix="/ai", tags=["AI Engine"])


@router.get("/health", tags=["System"])
def health_check() -> dict:
    return {"status": "ok", "service": "StockVision AI"}
