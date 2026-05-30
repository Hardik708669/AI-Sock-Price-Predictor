import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from app.ml.evaluation import confidence_from_metrics, regression_metrics
from app.ml.features import engineer_features
from app.ml.optional import optional_import


class StockPredictionEngine:
    def _split(self, X: pd.DataFrame, y: pd.Series):
        split = max(40, int(len(X) * 0.8))
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    def _classic_models(self) -> dict:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=220, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=220, learning_rate=0.045, max_depth=4, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42, verbosity=0),
        }

    def _prophet_forecast(self, data: pd.DataFrame) -> tuple[float, dict[str, float]] | None:
        Prophet = optional_import("prophet", "Prophet")
        if not Prophet:
            return None
        frame = data.reset_index()[[data.index.name or "Date", "Close"]]
        frame.columns = ["ds", "y"]
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(frame)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        predicted = float(forecast["yhat"].iloc[-1])
        validation = forecast["yhat"].iloc[-60:-1]
        actual = frame["y"].iloc[-59:]
        return predicted, regression_metrics(actual, validation.tail(len(actual)))

    def _lstm_forecast(self, data: pd.DataFrame) -> tuple[float, dict[str, float]] | None:
        tf = optional_import("tensorflow")
        if not tf or len(data) < 160:
            return self._lstm_fallback(data)
        close = data["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)
        window = 30
        X, y = [], []
        for i in range(window, len(scaled)):
            X.append(scaled[i - window : i])
            y.append(scaled[i])
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        split = int(len(X_arr) * 0.85)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(window, 1)),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_arr[:split], y_arr[:split], epochs=8, batch_size=16, verbose=0)
        test_pred = model.predict(X_arr[split:], verbose=0).reshape(-1, 1)
        latest = model.predict(X_arr[-1:].reshape(1, window, 1), verbose=0)
        predicted = float(scaler.inverse_transform(latest)[0][0])
        metrics = regression_metrics(scaler.inverse_transform(y_arr[split:]), scaler.inverse_transform(test_pred))
        return predicted, metrics

    def _lstm_fallback(self, data: pd.DataFrame) -> tuple[float, dict[str, float]]:
        closes = data["Close"]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(closes.values.reshape(-1, 1)).flatten()
        window = scaled[-21:]
        momentum = np.mean(np.diff(window))
        next_scaled = np.clip(window[-1] + momentum, 0, 1.25)
        predicted = float(scaler.inverse_transform([[next_scaled]])[0][0])
        naive_pred = closes.shift(1).dropna()
        return predicted, regression_metrics(closes.iloc[1:], naive_pred)

    def predict(self, data: pd.DataFrame, horizon: int = 1) -> dict:
        X, y = engineer_features(data, horizon=horizon)
        if len(X) < 80:
            raise ValueError("At least 80 clean market data rows are required for prediction")
        X_train, X_test, y_train, y_test = self._split(X, y)
        latest = X.tail(1)
        current_price = float(data["Close"].iloc[-1])
        results = []
        trained_models = {}
        for name, estimator in self._classic_models().items():
            estimator.fit(X_train, y_train)
            test_pred = estimator.predict(X_test)
            predicted = float(estimator.predict(latest)[0])
            metrics = regression_metrics(y_test, test_pred)
            trained_models[name] = estimator
            results.append({"model": name, "predicted_price": round(predicted, 2), "confidence": confidence_from_metrics(metrics, current_price, predicted), "metrics": metrics})
        prophet = self._prophet_forecast(data)
        if prophet:
            predicted, metrics = prophet
            results.append({"model": "Prophet", "predicted_price": round(predicted, 2), "confidence": confidence_from_metrics(metrics, current_price, predicted), "metrics": metrics})
        lstm_pred, lstm_metrics = self._lstm_forecast(data)
        results.append({"model": "LSTM" if optional_import("tensorflow") else "LSTM Momentum Fallback", "predicted_price": round(lstm_pred, 2), "confidence": confidence_from_metrics(lstm_metrics, current_price, lstm_pred), "metrics": lstm_metrics})
        weights = np.asarray([max(item["confidence"], 1) for item in results], dtype=float)
        predictions = np.asarray([item["predicted_price"] for item in results], dtype=float)
        ensemble = float(np.average(predictions, weights=weights))
        recommendation = "BUY" if ensemble > current_price * 1.025 else "SELL" if ensemble < current_price * 0.975 else "HOLD"
        return {"current_price": round(current_price, 2), "ensemble_prediction": round(ensemble, 2), "recommendation": recommendation, "models": results, "trained_models": trained_models, "feature_frame": X, "target": y}
