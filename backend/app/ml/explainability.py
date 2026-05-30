import numpy as np
import pandas as pd

from app.ml.optional import optional_import


class ExplainableAIEngine:
    def explain(self, model, X: pd.DataFrame) -> dict:
        latest = X.tail(1)
        shap = optional_import("shap")
        if shap:
            try:
                explainer = shap.Explainer(model, X.tail(250))
                values = explainer(latest)
                raw_values = values.values[0]
                importance = sorted(
                    [{"feature": feature, "importance": round(float(abs(value)), 6), "direction": "positive" if value >= 0 else "negative"} for feature, value in zip(X.columns, raw_values)],
                    key=lambda item: item["importance"],
                    reverse=True,
                )[:12]
                return {"method": "SHAP", "feature_importance": importance, "explanation": self._narrative(importance)}
            except Exception:
                pass
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.abs(model.coef_)
        else:
            values = latest.iloc[0].abs().to_numpy()
        importance = sorted(
            [{"feature": feature, "importance": round(float(abs(value)), 6), "direction": "positive"} for feature, value in zip(X.columns, values)],
            key=lambda item: item["importance"],
            reverse=True,
        )[:12]
        return {"method": "Model Feature Importance Fallback", "feature_importance": importance, "explanation": self._narrative(importance)}

    def _narrative(self, importance: list[dict]) -> str:
        top = importance[:3]
        factors = ", ".join(item["feature"].replace("_", " ") for item in top)
        return f"The prediction is primarily driven by {factors}. These factors describe recent momentum, volatility, trend alignment, and volume behavior."
