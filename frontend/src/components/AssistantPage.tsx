import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Bell, Bot, CalendarDays, Globe2, Mic, Send } from "lucide-react";
import { api } from "../lib/api";
import { Button } from "./ui/button";
import { Card, CardTitle } from "./ui/card";
import { Input } from "./ui/input";

export function AssistantPage() {
  const [message, setMessage] = useState("Should I buy Tesla?");
  const [alertSymbol, setAlertSymbol] = useState("AAPL");
  const assistant = useMutation({
    mutationFn: () => api<{ answer: string; used_data: string[] }>("/assistant", {
      method: "POST",
      body: JSON.stringify({ message, symbols: ["TSLA"] }),
    }),
  });
  const alert = useMutation({
    mutationFn: () => api<any>("/alerts", {
      method: "POST",
      body: JSON.stringify({ symbol: alertSymbol, metric: "price", operator: ">", threshold: 250, channel: "email" }),
    }),
  });

  return (
    <section className="mx-auto grid w-[min(1180px,calc(100%-28px))] gap-4 pb-24 lg:grid-cols-[1.3fr_.7fr]">
      <Card className="min-h-[620px]">
        <div className="flex items-center gap-3">
          <span className="grid h-11 w-11 place-items-center rounded-md bg-electric shadow-glow">
            <Bot />
          </span>
          <div>
            <CardTitle>AI Stock Assistant</CardTitle>
            <h1 className="text-2xl font-bold">Ask about risk, recommendations, comparisons, and predictions</h1>
          </div>
        </div>
        <div className="mt-6 rounded-lg border border-white/10 bg-white/5 p-5">
          <p className="text-sm text-slate-400">Example prompts</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {["Should I buy Tesla?", "What is the risk level of Nvidia?", "Compare Apple and Microsoft."].map((prompt) => (
              <button key={prompt} onClick={() => setMessage(prompt)} className="rounded-md bg-white/5 px-3 py-2 text-sm text-slate-300">
                {prompt}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-6 min-h-64 rounded-lg border border-white/10 bg-black/20 p-5">
          <p className="text-sm leading-7 text-slate-300">
            {assistant.data?.answer ?? "Ask a question to receive an AI answer using company data, sentiment data, prediction data, and technical signals."}
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {assistant.data?.used_data.map((item) => (
              <span key={item} className="rounded bg-emerald/10 px-2 py-1 text-xs text-emerald-200">{item}</span>
            ))}
          </div>
        </div>
        <div className="mt-5 flex gap-2">
          <Input value={message} onChange={(event) => setMessage(event.target.value)} className="flex-1" />
          <Button onClick={() => assistant.mutate()} className="gap-2">
            <Send size={16} />
            Ask
          </Button>
        </div>
      </Card>

      <div className="space-y-4">
        <Card>
          <CardTitle>Alert System</CardTitle>
          <p className="mt-3 text-sm text-slate-400">Create price and indicator alerts with email notifications.</p>
          <div className="mt-4 flex gap-2">
            <Input value={alertSymbol} onChange={(event) => setAlertSymbol(event.target.value.toUpperCase())} />
            <Button onClick={() => alert.mutate()} className="gap-2">
              <Bell size={16} />
              Create
            </Button>
          </div>
          <p className="mt-3 text-sm text-emerald">{alert.data?.message}</p>
        </Card>
        <Card>
          <CardTitle>Bonus Intelligence Modules</CardTitle>
          <div className="mt-4 grid gap-3 text-sm text-slate-300">
            <p className="rounded-md bg-white/5 p-3"><CalendarDays className="mr-2 inline" size={16} />Economic and earnings calendar</p>
            <p className="rounded-md bg-white/5 p-3"><Globe2 className="mr-2 inline" size={16} />Multi-language dashboard-ready architecture</p>
            <p className="rounded-md bg-white/5 p-3"><Mic className="mr-2 inline" size={16} />Voice assistant extension point</p>
          </div>
        </Card>
        <Card>
          <CardTitle>Stock Screener</CardTitle>
          <div className="mt-4 space-y-2 text-sm">
            {["AI Rating: BUY", "Volatility < 35%", "Market Sentiment: Bullish", "Volume Trend: Increasing"].map((item) => (
              <p key={item} className="rounded-md bg-white/5 p-3">{item}</p>
            ))}
          </div>
        </Card>
      </div>
    </section>
  );
}
