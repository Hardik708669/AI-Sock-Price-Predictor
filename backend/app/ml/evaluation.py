import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "mape": 0.0}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    denominator = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4), "mape": round(mape, 4)}


def confidence_from_metrics(metrics: dict[str, float], current_price: float, prediction: float) -> float:
    price_error_penalty = abs(prediction - current_price) / max(abs(current_price), 1) * 18
    mape_penalty = min(metrics.get("mape", 0.0), 60) * 0.7
    r2_bonus = max(metrics.get("r2", 0.0), 0) * 18
    return round(float(max(5, min(98, 82 - mape_penalty - price_error_penalty + r2_bonus))), 2)
