import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import ApexChart from "react-apexcharts";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, Candle, Prediction } from "../lib/api";
import { currency } from "../lib/utils";
import { useAppStore } from "../store/useAppStore";
import { Button } from "./ui/button";
import { Card, CardTitle } from "./ui/card";
import { Input } from "./ui/input";

const timeframes = ["1D", "5D", "1M", "6M", "1Y", "5Y", "MAX"];
const indicators = ["RSI", "MACD", "EMA", "SMA", "Bollinger Bands", "VWAP"];

export function StockAnalysis() {
  const { selectedSymbol, setSymbol } = useAppStore();
  const [query, setQuery] = useState(selectedSymbol);
  const [chartType, setChartType] = useState<"candlestick" | "line" | "area" | "volume">("candlestick");
  const [activeIndicators, setActiveIndicators] = useState(["RSI", "EMA"]);
  const candles = useQuery({
    queryKey: ["candles", selectedSymbol],
    queryFn: () => api<{ candles: Candle[] }>(`/stocks/${selectedSymbol}/candles?period=1y`),
  });
  const overview = useQuery({
    queryKey: ["overview", selectedSymbol],
    queryFn: () => api<any>(`/stocks/${selectedSymbol}/overview`),
  });
  const prediction = useQuery({
    queryKey: ["prediction", selectedSymbol],
    queryFn: () => api<Prediction>(`/predictions/${selectedSymbol}`),
  });
  const data = candles.data?.candles ?? [];
  const apexSeries = useMemo(
    () => [
      {
        data: data.map((item) => ({
          x: item.date,
          y: [item.open, item.high, item.low, item.close],
        })),
      },
    ],
    [data],
  );

  return (
    <section className="mx-auto w-[min(1180px,calc(100%-28px))] pb-24">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm uppercase text-electric">Stock Analysis</p>
          <h1 className="mt-2 text-4xl font-black">TradingView-style analysis for any stock</h1>
        </div>
        <div className="flex gap-2">
          <Input value={query} onChange={(event) => setQuery(event.target.value.toUpperCase())} placeholder="AAPL, TSLA, RELIANCE.NS" />
          <Button onClick={() => setSymbol(query)}>Analyze</Button>
        </div>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-[1.55fr_.45fr]">
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle>{selectedSymbol} Advanced Charting</CardTitle>
            <div className="flex flex-wrap gap-2">
              {(["candlestick", "line", "area", "volume"] as const).map((type) => (
                <button
                  key={type}
                  onClick={() => setChartType(type)}
                  className={`rounded-md px-3 py-1 text-xs capitalize ${chartType === type ? "bg-electric" : "bg-white/5 text-slate-400"}`}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            {timeframes.map((item) => <button key={item} className="rounded bg-white/5 px-3 py-1 text-xs text-slate-300">{item}</button>)}
          </div>
          <div className="mt-5 h-[420px]">
            {chartType === "candlestick" ? (
              <ApexChart
                type="candlestick"
                height="100%"
                series={apexSeries}
                options={{
                  theme: { mode: "dark" },
                  chart: { background: "transparent", toolbar: { show: false } },
                  xaxis: { labels: { style: { colors: "#94a3b8" } } },
                  yaxis: { labels: { style: { colors: "#94a3b8" } } },
                  grid: { borderColor: "rgba(148,163,184,.15)" },
                }}
              />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                {chartType === "volume" ? (
                  <BarChart data={data}>
                    <CartesianGrid stroke="rgba(148,163,184,.12)" />
                    <XAxis dataKey="date" hide />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="volume" fill="#2f7cff" />
                  </BarChart>
                ) : (
                  <AreaChart data={data}>
                    <CartesianGrid stroke="rgba(148,163,184,.12)" />
                    <XAxis dataKey="date" hide />
                    <YAxis />
                    <Tooltip />
                    <Area dataKey="close" stroke="#34d399" fill="#10b98122" />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            )}
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            {indicators.map((item) => (
              <button
                key={item}
                onClick={() =>
                  setActiveIndicators((current) => current.includes(item) ? current.filter((active) => active !== item) : [...current, item])
                }
                className={`rounded-md px-3 py-1 text-xs ${activeIndicators.includes(item) ? "bg-emerald/20 text-emerald-200" : "bg-white/5 text-slate-400"}`}
              >
                {item}
              </button>
            ))}
          </div>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardTitle>Company Overview</CardTitle>
            <h2 className="mt-3 text-2xl font-bold">{overview.data?.name ?? selectedSymbol}</h2>
            <p className="mt-2 text-3xl font-black text-emerald">
              {currency(overview.data?.live_price ?? 0, overview.data?.currency ?? "USD")}
            </p>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-slate-300">
              <span>Market Cap: {overview.data?.market_cap ?? "N/A"}</span>
              <span>Volume: {overview.data?.volume ?? "N/A"}</span>
              <span>PE Ratio: {overview.data?.pe_ratio ?? "N/A"}</span>
              <span>EPS: {overview.data?.eps ?? "N/A"}</span>
              <span>Dividend: {overview.data?.dividend_yield ?? "N/A"}</span>
            </div>
          </Card>
          <Card>
            <CardTitle>AI Buy/Sell Engine</CardTitle>
            <p className="mt-4 text-5xl font-black gradient-text">{prediction.data?.recommendation ?? "HOLD"}</p>
            <p className="mt-4 text-sm leading-6 text-slate-300">{prediction.data?.explanation ?? "Run analysis to generate a recommendation."}</p>
          </Card>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardTitle>Multi-Model AI Prediction Engine</CardTitle>
          <div className="mt-4 space-y-3">
            {prediction.data?.models.map((model) => (
              <div key={model.model_name} className="grid grid-cols-4 gap-2 rounded-md bg-white/5 p-3 text-sm">
                <span>{model.model_name}</span>
                <span>{currency(model.predicted_price)}</span>
                <span>{model.confidence}% confidence</span>
                <span>{model.accuracy}% accuracy</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <CardTitle>Explainable AI Feature Importance</CardTitle>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={prediction.data?.feature_importance ?? []}>
              <XAxis dataKey="feature" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="importance" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </section>
  );
}
