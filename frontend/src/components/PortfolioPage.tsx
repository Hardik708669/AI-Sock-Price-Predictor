import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Pie, PieChart, Cell, ResponsiveContainer, Tooltip, Treemap } from "recharts";
import { api } from "../lib/api";
import { currency, percent } from "../lib/utils";
import { Button } from "./ui/button";
import { Card, CardTitle } from "./ui/card";
import { Input } from "./ui/input";

const sampleHoldings = [
  { symbol: "AAPL", quantity: 12, average_price: 182, current_price: 212.44, sector: "Technology" },
  { symbol: "NVDA", quantity: 18, average_price: 116, current_price: 139.89, sector: "Technology" },
  { symbol: "JPM", quantity: 9, average_price: 198, current_price: 214.3, sector: "Banking" },
  { symbol: "JNJ", quantity: 10, average_price: 151, current_price: 147.2, sector: "Healthcare" },
  { symbol: "XOM", quantity: 14, average_price: 108, current_price: 113.9, sector: "Energy" },
];

export function PortfolioPage() {
  const [symbol, setSymbol] = useState("MSFT");
  const [holdings, setHoldings] = useState(sampleHoldings);
  const portfolio = useMutation({
    mutationFn: () => api<any>("/portfolio/analyze", { method: "POST", body: JSON.stringify({ holdings }) }),
  });
  const heatmap = useQuery({ queryKey: ["heatmap"], queryFn: () => api<any>("/heatmap") });
  const colors = ["#2f7cff", "#8b5cf6", "#10b981", "#f97316", "#ef4444"];
  const heatmapData = useMemo(
    () => ({
      name: "Market",
      children: (heatmap.data?.categories ?? []).map((item: any) => ({
        name: item.sector,
        size: Math.abs(item.change) * 100,
        change: item.change,
      })),
    }),
    [heatmap.data],
  );

  function addHolding() {
    setHoldings((current) => [
      ...current,
      { symbol: symbol.toUpperCase(), quantity: 5, average_price: 100, current_price: 108, sector: "Technology" },
    ]);
  }

  return (
    <section className="mx-auto w-[min(1180px,calc(100%-28px))] pb-24">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm uppercase text-electric">Portfolio Management</p>
          <h1 className="mt-2 text-4xl font-black">Health score, risk score, allocation, and auto rebalance</h1>
        </div>
        <div className="flex gap-2">
          <Input value={symbol} onChange={(event) => setSymbol(event.target.value)} />
          <Button onClick={addHolding}>Add Stock</Button>
          <Button variant="ghost" onClick={() => portfolio.mutate()}>Optimize</Button>
        </div>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-[1.2fr_.8fr]">
        <Card>
          <CardTitle>Holdings</CardTitle>
          <div className="mt-4 space-y-3">
            {holdings.map((holding) => (
              <div key={holding.symbol} className="grid grid-cols-5 gap-2 rounded-md bg-white/5 p-3 text-sm">
                <span className="font-bold">{holding.symbol}</span>
                <span>{holding.quantity} shares</span>
                <span>Avg {currency(holding.average_price)}</span>
                <span>Now {currency(holding.current_price)}</span>
                <button onClick={() => setHoldings((items) => items.filter((item) => item.symbol !== holding.symbol))} className="text-danger">
                  Remove
                </button>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <CardTitle>Portfolio Health</CardTitle>
          <div className="mt-5 grid grid-cols-2 gap-3">
            <div className="rounded-md bg-white/5 p-4">
              <p className="text-sm text-slate-400">Current Value</p>
              <p className="mt-2 text-2xl font-bold">{currency(portfolio.data?.current_value ?? 0)}</p>
            </div>
            <div className="rounded-md bg-white/5 p-4">
              <p className="text-sm text-slate-400">Profit/Loss</p>
              <p className="mt-2 text-2xl font-bold text-emerald">{currency(portfolio.data?.profit_loss ?? 0)}</p>
            </div>
            <div className="rounded-md bg-white/5 p-4">
              <p className="text-sm text-slate-400">Health Score</p>
              <p className="mt-2 text-2xl font-bold">{portfolio.data?.portfolio_health_score ?? "--"}/100</p>
            </div>
            <div className="rounded-md bg-white/5 p-4">
              <p className="text-sm text-slate-400">Risk Score</p>
              <p className="mt-2 text-2xl font-bold text-purple">{portfolio.data?.risk_score ?? "--"}/100</p>
            </div>
          </div>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card>
          <CardTitle>Asset Allocation</CardTitle>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={portfolio.data?.allocation ?? []} dataKey="value" nameKey="name" outerRadius={92}>
                {(portfolio.data?.allocation ?? []).map((_entry: any, index: number) => <Cell key={index} fill={colors[index % colors.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
        <Card>
          <CardTitle>Auto Rebalance</CardTitle>
          <div className="mt-4 space-y-3 text-sm text-slate-300">
            {(portfolio.data?.rebalance_actions ?? ["Click Optimize to generate rebalance actions."]).map((item: string) => (
              <p key={item} className="rounded-md bg-white/5 p-3">{item}</p>
            ))}
          </div>
        </Card>
        <Card>
          <CardTitle>Market Heatmap</CardTitle>
          <ResponsiveContainer width="100%" height={250}>
            <Treemap data={heatmapData.children} dataKey="size" nameKey="name" stroke="#050712" fill="#10b981" />
          </ResponsiveContainer>
          <div className="mt-2 flex flex-wrap gap-2 text-xs">
            {(heatmap.data?.categories ?? []).map((item: any) => (
              <span key={item.sector} className={item.change >= 0 ? "text-emerald" : "text-danger"}>
                {item.sector} {percent(item.change)}
              </span>
            ))}
          </div>
        </Card>
      </div>
    </section>
  );
}
