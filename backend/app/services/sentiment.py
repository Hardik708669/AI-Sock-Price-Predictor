from datetime import datetime, timezone

from app.schemas import SentimentItem


class SentimentService:
    KEYWORDS = {
        "Bullish": ["beat", "growth", "upgrade", "record", "profit", "expands", "surge"],
        "Bearish": ["miss", "lawsuit", "downgrade", "fall", "loss", "cuts", "probe"],
    }

    def latest(self, symbol: str) -> list[SentimentItem]:
        samples = [
            f"{symbol.upper()} posts resilient demand as analysts lift revenue outlook",
            f"Options volume rises while traders wait for {symbol.upper()} earnings",
            f"{symbol.upper()} faces margin questions after sector-wide volatility",
        ]
        items = []
        for title in samples:
            lowered = title.lower()
            sentiment = "Neutral"
            score = 0.04
            for label, keywords in self.KEYWORDS.items():
                hits = sum(keyword in lowered for keyword in keywords)
                if hits:
                    sentiment = label
                    score = 0.35 if label == "Bullish" else -0.35
            items.append(
                SentimentItem(
                    title=title,
                    source="StockVision News Engine",
                    sentiment=sentiment,
                    score=score,
                    summary=f"AI summary: {title}. The signal is classified as {sentiment.lower()} for short-term sentiment.",
                    published_at=datetime.now(timezone.utc),
                )
            )
        return items

    def aggregate(self, symbol: str) -> dict:
        news = self.latest(symbol)
        score = sum(item.score for item in news) / len(news)
        label = "Bullish" if score > 0.12 else "Bearish" if score < -0.12 else "Neutral"
        return {
            "symbol": symbol.upper(),
            "sentiment": label,
            "score": round(score, 3),
            "trend": [round(score * factor, 3) for factor in [0.45, 0.7, 0.95, 1.0]],
            "news": news,
        }
