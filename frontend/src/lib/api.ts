const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<T>;
}

export type DashboardData = {
  portfolio_value: number;
  daily_profit_loss: number;
  ai_confidence_score: number;
  market_sentiment: string;
  prediction_center: Array<{ symbol: string; signal: string; confidence: number }>;
};

export type Prediction = {
  symbol: string;
  current_price: number;
  recommendation: "BUY" | "SELL" | "HOLD";
  explanation: string;
  models: Array<{ model_name: string; predicted_price: number; confidence: number; accuracy: number }>;
  feature_importance: Array<{ feature: string; importance: number }>;
};

export type Candle = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};
