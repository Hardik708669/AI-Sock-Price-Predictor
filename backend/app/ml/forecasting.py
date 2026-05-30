import pandas as pd

from app.ml.prediction_engine import StockPredictionEngine


class ForecastingEngine:
    horizons = [1, 7, 30, 90]

    def forecast(self, data: pd.DataFrame) -> dict:
        engine = StockPredictionEngine()
        forecasts = {}
        for horizon in self.horizons:
            try:
                result = engine.predict(data, horizon=horizon)
                forecasts[f"{horizon}_day"] = {
                    "predicted_price": result["ensemble_prediction"],
                    "recommendation": result["recommendation"],
                    "model_count": len(result["models"]),
                }
            except Exception as error:
                forecasts[f"{horizon}_day"] = {"error": str(error)}
        return forecasts
