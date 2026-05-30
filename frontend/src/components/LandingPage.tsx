import { motion } from "framer-motion";
import { Activity, Bot, Brain, Briefcase, Gauge, ShieldCheck } from "lucide-react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { Button } from "./ui/button";
import { Card } from "./ui/card";

const chartData = Array.from({ length: 64 }, (_, index) => ({
  index,
  value: 120 + Math.sin(index / 4) * 14 + index * 1.4 + (index % 9) * 2,
}));

const features = [
  { icon: Brain, title: "AI Predictions", copy: "Linear Regression, Random Forest, XGBoost, trend forecast, and LSTM-style momentum signals." },
  { icon: Activity, title: "Sentiment Analysis", copy: "News classification, sentiment trend, and AI market summaries." },
  { icon: Briefcase, title: "Portfolio Tracker", copy: "Holdings, allocation, health score, P/L, and automatic rebalancing." },
  { icon: Gauge, title: "Risk Analysis", copy: "Volatility, beta, Sharpe ratio, drawdown, and risk tiers." },
  { icon: Bot, title: "AI Assistant", copy: "Ask questions about stocks using prediction, sentiment, and company data." },
  { icon: ShieldCheck, title: "Secure SaaS Core", copy: "Firebase auth, JWT, validation, rate limiting, and PostgreSQL schema." },
];

export function LandingPage({ openDashboard }: { openDashboard: () => void }) {
  return (
    <div>
      <section className="relative mx-auto flex min-h-[calc(100vh-96px)] w-[min(1180px,calc(100%-28px))] items-center overflow-hidden rounded-lg border border-white/10 bg-black/20 px-5 py-12">
        <div className="absolute inset-0 chart-grid bg-grid opacity-50" />
        <div className="absolute inset-x-0 bottom-0 h-[56%] opacity-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="heroChart" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#2f7cff" stopOpacity={0.8} />
                  <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.04} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="value" stroke="#60a5fa" strokeWidth={3} fill="url(#heroChart)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="relative z-10 max-w-3xl"
        >
          <p className="mb-4 inline-flex rounded-md border border-emerald/30 bg-emerald/10 px-3 py-1 text-sm text-emerald-200">
            Startup-grade AI stock intelligence platform
          </p>
          <h1 className="text-5xl font-black leading-tight tracking-normal text-white md:text-7xl">
            StockVision AI
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-300">
            A full-stack fintech dashboard for predictions, charting, sentiment, portfolios, risk, alerts, and an AI stock assistant.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button onClick={openDashboard}>Launch Dashboard</Button>
            <Button variant="ghost">View Demo</Button>
          </div>
          <div className="mt-10 grid grid-cols-3 gap-3">
            {[
              ["5", "AI Models"],
              ["84%", "AI Confidence"],
              ["24/7", "Market Watch"],
            ].map(([value, label]) => (
              <Card key={label} className="p-4">
                <p className="text-3xl font-bold gradient-text">{value}</p>
                <p className="mt-1 text-xs text-slate-400">{label}</p>
              </Card>
            ))}
          </div>
        </motion.div>
      </section>

      <section className="mx-auto grid w-[min(1180px,calc(100%-28px))] gap-4 py-14 md:grid-cols-3">
        {features.map((feature, index) => {
          const Icon = feature.icon;
          return (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.05 }}
            >
              <Card className="h-full">
                <Icon className="text-electric" />
                <h3 className="mt-4 text-xl font-bold">{feature.title}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-400">{feature.copy}</p>
              </Card>
            </motion.div>
          );
        })}
      </section>

      <section className="mx-auto grid w-[min(1180px,calc(100%-28px))] gap-4 py-8 lg:grid-cols-[1.3fr_.7fr]">
        <Card>
          <p className="text-sm uppercase text-slate-400">Interactive Demo</p>
          <h2 className="mt-2 text-3xl font-bold">Prediction Center Preview</h2>
          <div className="mt-6 grid gap-3 md:grid-cols-3">
            {["BUY AAPL", "HOLD TSLA", "BUY NVDA"].map((signal) => (
              <div key={signal} className="rounded-md border border-white/10 bg-white/5 p-4">
                <p className="text-lg font-bold">{signal}</p>
                <p className="mt-1 text-sm text-slate-400">Confidence score, trend strength, sentiment, and model comparison.</p>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <p className="text-sm uppercase text-slate-400">Pricing</p>
          <h2 className="mt-2 text-3xl font-bold">$0 MVP</h2>
          <p className="mt-3 text-sm leading-6 text-slate-400">Perfect for portfolio projects, hackathons, demos, and internship interviews.</p>
        </Card>
      </section>

      <footer className="mx-auto w-[min(1180px,calc(100%-28px))] py-10 text-sm text-slate-500">
        StockVision AI - Built with React, TypeScript, FastAPI, PostgreSQL, Firebase, and multi-model ML.
      </footer>
    </div>
  );
}
