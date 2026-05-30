from app.jobs.worker import celery_app
from app.ml.training_pipeline import TrainingPipeline
from app.services.market_data import get_market_data_service


@celery_app.task(name="market.refresh_stock_snapshot")
def refresh_stock_snapshot(ticker: str) -> dict:
    overview = get_market_data_service().overview(ticker)
    return overview.model_dump()


@celery_app.task(name="alerts.evaluate")
def evaluate_alerts() -> dict:
    return {"evaluated": True}


@celery_app.task(name="models.retrain_stock")
def retrain_stock_model(ticker: str, period: str = "5y") -> dict:
    return TrainingPipeline().train_and_register(ticker, period)
