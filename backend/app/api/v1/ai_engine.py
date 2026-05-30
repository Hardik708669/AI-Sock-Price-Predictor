from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.deps import get_current_user
from app.ml.engine import StockVisionAIEngine, get_ai_engine
from app.ml.training_pipeline import TrainingPipeline
from app.models.domain import User
from app.schemas import PortfolioOptimizeRequest, RetrainRequest

router = APIRouter()


@router.get("/intelligence/{ticker}")
def intelligence_report(ticker: str, _user: User = Depends(get_current_user), engine: StockVisionAIEngine = Depends(get_ai_engine)) -> dict:
    try:
        return engine.intelligence_report(ticker)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/technical-analysis/{ticker}")
def technical_analysis(ticker: str, _user: User = Depends(get_current_user), engine: StockVisionAIEngine = Depends(get_ai_engine)) -> dict:
    data = engine.data.historical_prices(ticker)
    return engine.technicals.signal(data)


@router.get("/sentiment/{ticker}")
def sentiment(ticker: str, _user: User = Depends(get_current_user), engine: StockVisionAIEngine = Depends(get_ai_engine)) -> dict:
    return engine.sentiment.analyze_news(engine.data.financial_news(ticker))


@router.get("/trend/{ticker}")
def trend(ticker: str, _user: User = Depends(get_current_user), engine: StockVisionAIEngine = Depends(get_ai_engine)) -> dict:
    return engine.trends.detect(engine.data.historical_prices(ticker))


@router.get("/risk/{ticker}")
def risk(ticker: str, _user: User = Depends(get_current_user), engine: StockVisionAIEngine = Depends(get_ai_engine)) -> dict:
    return engine.risk.analyze(engine.data.historical_prices(ticker), engine.data.historical_prices("^GSPC"))


@router.get("/anomalies/{ticker}")
def anomalies(ticker: str, _user: User = Depends(get_current_user), engine: StockVisionAIEngine = Depends(get_ai_engine)) -> dict:
    return engine.anomalies.detect(engine.data.historical_prices(ticker))


@router.get("/forecast/{ticker}")
def forecast(ticker: str, _user: User = Depends(get_current_user), engine: StockVisionAIEngine = Depends(get_ai_engine)) -> dict:
    return engine.forecasting.forecast(engine.data.historical_prices(ticker))


@router.post("/portfolio/optimize")
def optimize(payload: PortfolioOptimizeRequest, _user: User = Depends(get_current_user), engine: StockVisionAIEngine = Depends(get_ai_engine)) -> dict:
    try:
        return engine.optimize_portfolio(payload.tickers, payload.investment_amount, payload.risk_appetite)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/retrain")
def retrain(payload: RetrainRequest, background_tasks: BackgroundTasks, _user: User = Depends(get_current_user)) -> dict:
    def run_training() -> None:
        TrainingPipeline().train_and_register(payload.ticker, payload.period)

    background_tasks.add_task(run_training)
    return {"message": "Retraining scheduled", "ticker": payload.ticker.upper(), "period": payload.period}
