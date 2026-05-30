from app.ml.data_pipeline import MarketDataPipeline
from app.ml.prediction_engine import StockPredictionEngine
from app.ml.registry import ModelRegistry


class TrainingPipeline:
    def __init__(self, data: MarketDataPipeline | None = None, registry: ModelRegistry | None = None) -> None:
        self.data = data or MarketDataPipeline()
        self.registry = registry or ModelRegistry()

    def train_and_register(self, ticker: str, period: str = "5y") -> dict:
        frame = self.data.historical_prices(ticker, period=period)
        result = StockPredictionEngine().predict(frame)
        records = []
        for name, model in result["trained_models"].items():
            model_metrics = next(item["metrics"] for item in result["models"] if item["model"] == name)
            records.append(self.registry.register(name=f"{ticker.upper()} {name}", model=model, metrics=model_metrics, metadata={"ticker": ticker.upper()}))
        return {"ticker": ticker.upper(), "registered_models": records, "ensemble_prediction": result["ensemble_prediction"], "recommendation": result["recommendation"]}
