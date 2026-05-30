import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { EyeOff, GripVertical } from "lucide-react";
import { api, DashboardData } from "../lib/api";
import { currency } from "../lib/utils";
import { useAppStore, WidgetKey } from "../store/useAppStore";
import { Card, CardTitle, Metric } from "./ui/card";
import { Button } from "./ui/button";

const labels: Record<WidgetKey, string> = {
  portfolio: "Portfolio Value",
  profitLoss: "Daily Profit/Loss",
  confidence: "AI Confidence",
  sentiment: "Market Sentiment",
  watchlist: "Watchlist",
  news: "News Feed",
  prediction: "Prediction Center",
  trending: "Trending Stocks",
};

export function Dashboard() {
  const { data } = useQuery({ queryKey: ["dashboard"], queryFn: () => api<DashboardData>("/dashboard") });
  const { widgets, hiddenWidgets, toggleWidget, moveWidget } = useAppStore();
  const dashboard = data ?? {
    portfolio_value: 128940.55,
    daily_profit_loss: 1842.2,
    ai_confidence_score: 84,
    market_sentiment: "Bullish",
    prediction_center: [],
  };

  function renderWidget(widget: WidgetKey, index: number) {
    if (hiddenWidgets.includes(widget)) return null;
    if (widget === "portfolio") return <Metric label={labels[widget]} value={currency(dashboard.portfolio_value)} accent="text-emerald" />;
    if (widget === "profitLoss") return <Metric label={labels[widget]} value={currency(dashboard.daily_profit_loss)} accent="text-emerald" />;
    if (widget === "confidence") return <Metric label={labels[widget]} value={`${dashboard.ai_confidence_score}/100`} accent="text-electric" />;
    if (widget === "sentiment") return <Metric label={labels[widget]} value={dashboard.market_sentiment} accent="text-purple" />;
    return (
      <Card className="min-h-48">
        <div className="flex items-center justify-between">
          <CardTitle>{labels[widget]}</CardTitle>
          <div className="flex gap-1">
            <button onClick={() => index > 0 && moveWidget(index, index - 1)} className="rounded p-1 text-slate-500 hover:bg-white/10">
              <GripVertical size={16} />
            </button>
            <button onClick={() => toggleWidget(widget)} className="rounded p-1 text-slate-500 hover:bg-white/10">
              <EyeOff size={16} />
            </button>
          </div>
        </div>
        {widget === "prediction" && (
          <div className="mt-4 space-y-3">
            {dashboard.prediction_center.map((item) => (
              <div key={item.symbol} className="flex items-center justify-between rounded-md bg-white/5 p-3">
                <span>{item.symbol}</span>
                <span className="text-emerald">{item.signal}</span>
                <span>{item.confidence}%</span>
              </div>
            ))}
          </div>
        )}
        {widget === "watchlist" && <p className="mt-5 text-sm text-slate-400">AAPL, NVDA, RELIANCE.NS with price, daily change, and AI rating.</p>}
        {widget === "news" && <p className="mt-5 text-sm text-slate-400">Latest AI-classified financial news with bullish, bearish, and neutral tags.</p>}
      </Card>
    );
  }

  return (
    <section className="mx-auto w-[min(1180px,calc(100%-28px))] pb-24">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-sm uppercase text-electric">Main Dashboard</p>
          <h1 className="mt-2 text-4xl font-black">Customizable AI trading command center</h1>
        </div>
        <Button variant="ghost">Save Layout</Button>
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {widgets.slice(0, 4).map((widget, index) => (
          <motion.div key={widget} layout>{renderWidget(widget, index)}</motion.div>
        ))}
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        {widgets.slice(4).map((widget, index) => (
          <motion.div key={widget} layout>{renderWidget(widget, index + 4)}</motion.div>
        ))}
      </div>
    </section>
  );
}
