from statistics import mean

from app.core.config import get_settings
from app.ml.data_pipeline import NewsArticle
from app.ml.optional import optional_import


class FinancialSentimentEngine:
    bullish_words = {"beat", "growth", "upgrade", "record", "profit", "surge", "expands", "strong", "raises", "outperform"}
    bearish_words = {"miss", "lawsuit", "downgrade", "fall", "loss", "cuts", "probe", "weak", "risk", "decline"}

    def __init__(self) -> None:
        self.transformer = None
        self.transformer_checked = False

    def _get_transformer(self):
        if self.transformer_checked or not get_settings().enable_transformer_sentiment:
            return self.transformer
        self.transformer_checked = True
        pipeline = optional_import("transformers", "pipeline")
        if pipeline:
            try:
                self.transformer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            except Exception:
                self.transformer = None
        return self.transformer

    def classify_text(self, text: str) -> dict:
        transformer = self._get_transformer()
        if transformer:
            result = transformer(text[:512])[0]
            label = str(result["label"]).lower()
            score = float(result["score"])
            if "positive" in label:
                return {"sentiment": "Bullish", "score": round(score, 4)}
            if "negative" in label:
                return {"sentiment": "Bearish", "score": round(-score, 4)}
            return {"sentiment": "Neutral", "score": 0.0}
        lowered = text.lower()
        bull = sum(word in lowered for word in self.bullish_words)
        bear = sum(word in lowered for word in self.bearish_words)
        raw = (bull - bear) / max(bull + bear, 1)
        sentiment = "Bullish" if raw > 0.15 else "Bearish" if raw < -0.15 else "Neutral"
        return {"sentiment": sentiment, "score": round(float(raw), 4)}

    def analyze_news(self, articles: list[NewsArticle]) -> dict:
        analyzed = []
        for article in articles:
            result = self.classify_text(f"{article.title}. {article.summary}")
            analyzed.append({**result, "title": article.title, "source": article.source, "url": article.url, "summary": article.summary, "published_at": article.published_at})
        aggregate = mean([item["score"] for item in analyzed]) if analyzed else 0.0
        label = "Bullish" if aggregate > 0.15 else "Bearish" if aggregate < -0.15 else "Neutral"
        return {"sentiment": label, "score": round(float(aggregate), 4), "articles": analyzed}
