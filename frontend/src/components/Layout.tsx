import type { ReactNode } from "react";
import { BarChart3, Bot, Briefcase, Home, LineChart, LogIn, Radar, Search, Shield } from "lucide-react";
import { Button } from "./ui/button";

type View = "landing" | "auth" | "dashboard" | "analysis" | "portfolio" | "assistant";

const navItems = [
  { id: "landing", label: "Home", icon: Home },
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "analysis", label: "Analysis", icon: LineChart },
  { id: "portfolio", label: "Portfolio", icon: Briefcase },
  { id: "assistant", label: "Assistant", icon: Bot },
] as const;

export function Layout({
  view,
  setView,
  children,
}: {
  view: View;
  setView: (view: View) => void;
  children: ReactNode;
}) {
  return (
    <div className="min-h-screen">
      <nav className="fixed left-1/2 top-4 z-50 flex w-[min(1120px,calc(100%-24px))] -translate-x-1/2 items-center justify-between rounded-lg border border-white/10 bg-navy/80 px-3 py-2 backdrop-blur-2xl">
        <button className="flex items-center gap-2 px-2 text-left" onClick={() => setView("landing")}>
          <span className="grid h-8 w-8 place-items-center rounded-md bg-electric shadow-glow">
            <Radar size={18} />
          </span>
          <span>
            <span className="block text-sm font-bold">StockVision AI</span>
            <span className="block text-[11px] text-slate-400">AI Stock Intelligence</span>
          </span>
        </button>
        <div className="hidden items-center gap-1 md:flex">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm transition ${
                  view === item.id ? "bg-white/10 text-white" : "text-slate-400 hover:bg-white/5 hover:text-white"
                }`}
              >
                <Icon size={16} />
                {item.label}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" className="hidden gap-2 sm:flex" onClick={() => setView("analysis")}>
            <Search size={16} />
            Search
          </Button>
          <Button className="gap-2" onClick={() => setView("auth")}>
            <LogIn size={16} />
            Login
          </Button>
        </div>
      </nav>
      <main className="pt-24">{children}</main>
      <div className="fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 gap-1 rounded-lg border border-white/10 bg-navy/90 p-1 backdrop-blur-xl md:hidden">
        {navItems.slice(1).map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => setView(item.id)}
              className={`rounded-md p-3 ${view === item.id ? "bg-white/10 text-white" : "text-slate-400"}`}
              aria-label={item.label}
            >
              <Icon size={18} />
            </button>
          );
        })}
      </div>
      <div className="fixed bottom-4 right-4 hidden rounded-md border border-emerald/30 bg-emerald/10 px-3 py-2 text-xs text-emerald-200 lg:block">
        <Shield className="mr-2 inline" size={14} />
        JWT, Firebase Auth, rate limiting, validation
      </div>
    </div>
  );
}
