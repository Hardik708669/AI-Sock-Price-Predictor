import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetectionEngine:
    def detect(self, data: pd.DataFrame) -> dict:
        frame = pd.DataFrame(index=data.index)
        frame["return"] = data["Close"].pct_change()
        frame["volume_zscore"] = (data["Volume"] - data["Volume"].rolling(20).mean()) / data["Volume"].rolling(20).std()
        frame["range"] = (data["High"] - data["Low"]) / data["Close"]
        frame = frame.replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(frame) < 30:
            return {"anomalies": [], "latest_anomaly": False, "risk_flags": []}
        model = IsolationForest(contamination=0.04, random_state=42)
        labels = model.fit_predict(frame)
        scores = model.decision_function(frame)
        anomalies = []
        for index, label, score in zip(frame.index, labels, scores):
            if label == -1:
                row = frame.loc[index]
                flags = []
                if abs(row["return"]) > frame["return"].std() * 2.5:
                    flags.append("Sudden Price Spike" if row["return"] > 0 else "Sudden Price Drop")
                if row["volume_zscore"] > 2.5:
                    flags.append("Abnormal Volume")
                if row["range"] > frame["range"].quantile(0.95):
                    flags.append("Possible Market Manipulation")
                anomalies.append({"date": index.strftime("%Y-%m-%d"), "score": round(float(score), 4), "flags": flags or ["Statistical Anomaly"]})
        return {"anomalies": anomalies[-20:], "latest_anomaly": bool(labels[-1] == -1), "risk_flags": anomalies[-1]["flags"] if anomalies else []}
