import * as React from "react";
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  Pie,
  PieChart,
  Radar,
  RadarChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Treemap,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  ArrowRight,
  BadgeDollarSign,
  Bell,
  Bot,
  Brain,
  BriefcaseBusiness,
  CandlestickChart,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleDollarSign,
  ClipboardList,
  Command,
  Compass,
  Eye,
  Gauge,
  Grid3X3,
  History,
  Landmark,
  LayoutDashboard,
  Lock,
  Mail,
  Menu,
  MessageSquareText,
  Newspaper,
  Radar as RadarIcon,
  Search,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  UserPlus,
  WalletCards,
  X,
} from "lucide-react";
import {
  Link,
  NavLink,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { cn, currency, percent } from "./lib/utils";
import { useAppStore } from "./store/useAppStore";

const chartData = Array.from({ length: 72 }, (_, index) => ({
  day: `D${index + 1}`,
  price: 142 + index * 1.6 + Math.sin(index / 3.2) * 14 + (index % 7) * 2,
  ai: 148 + index * 1.45 + Math.cos(index / 4) * 11,
  volume: 42 + Math.abs(Math.sin(index / 5)) * 70 + (index % 8) * 5,
  sentiment: 48 + Math.sin(index / 6) * 32,
}));

const candles = Array.from({ length: 34 }, (_, index) => {
  const base = 172 + index * 2 + Math.sin(index / 2) * 9;
  const open = base + Math.sin(index) * 5;
  const close = base + Math.cos(index / 1.4) * 6;
  return {
    day: `${index + 1}`,
    low: Math.min(open, close) - 7 - (index % 3),
    high: Math.max(open, close) + 7 + (index % 4),
    open,
    close,
  };
});

const stocks = [
  { symbol: "NVDA", name: "NVIDIA", price: 139.89, change: 4.82, confidence: 92, sentiment: "Euphoric", sector: "AI Semis" },
  { symbol: "AAPL", name: "Apple", price: 212.44, change: 1.21, confidence: 84, sentiment: "Bullish", sector: "Consumer Tech" },
  { symbol: "MSFT", name: "Microsoft", price: 438.12, change: 2.06, confidence: 88, sentiment: "Bullish", sector: "Cloud" },
  { symbol: "TSLA", name: "Tesla", price: 183.92, change: -1.63, confidence: 61, sentiment: "Mixed", sector: "EV" },
  { symbol: "JPM", name: "JPMorgan", price: 214.3, change: 0.74, confidence: 72, sentiment: "Stable", sector: "Banking" },
];

const holdings = [
  { name: "AI Semis", value: 36, color: "#3B82F6" },
  { name: "Cloud", value: 24, color: "#8B5CF6" },
  { name: "Consumer", value: 18, color: "#06B6D4" },
  { name: "Energy", value: 12, color: "#10B981" },
  { name: "Cash", value: 10, color: "#64748B" },
];

const heatmap = [
  { name: "NVDA", size: 52, change: 4.8 },
  { name: "MSFT", size: 46, change: 2.1 },
  { name: "AAPL", size: 39, change: 1.2 },
  { name: "TSLA", size: 32, change: -1.6 },
  { name: "AMZN", size: 35, change: 0.8 },
  { name: "XOM", size: 25, change: -0.5 },
  { name: "JPM", size: 28, change: 0.7 },
  { name: "META", size: 31, change: 2.7 },
];

const news = [
  "AI chip demand pushes semiconductor guidance above consensus.",
  "Cloud capex remains elevated as enterprise copilots reach production.",
  "Fed minutes show measured optimism around inflation normalization.",
  "Energy equities soften after inventory data surprises to upside.",
];

const heroFeatures: Array<{ title: string; copy: string; icon: React.ElementType }> = [
  { title: "Neural Forecasts", copy: "Multi-model prediction scores with confidence ranges and factor attribution.", icon: Brain },
  { title: "Live Market Surface", copy: "Heatmaps, watchlists, sentiment, and portfolio drift in one command center.", icon: Activity },
  { title: "Risk-Aware Copilot", copy: "Conversational workflows that understand holdings, exposure, and goals.", icon: Bot },
];

const marketingPages = [
  { path: "/", label: "Home" },
  { path: "/about", label: "About" },
  { path: "/features", label: "Features" },
  { path: "/pricing", label: "Pricing" },
  { path: "/contact", label: "Contact" },
];

const appNav = [
  { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { path: "/stock-analysis", label: "Stock Analysis", icon: Search },
  { path: "/advanced-chart", label: "Advanced Chart", icon: CandlestickChart },
  { path: "/ai-prediction", label: "AI Prediction", icon: Brain },
  { path: "/explainable-ai", label: "Explainable AI", icon: Eye },
  { path: "/portfolio", label: "Portfolio", icon: WalletCards },
  { path: "/optimizer", label: "Optimizer", icon: Sparkles },
  { path: "/transactions", label: "Transactions", icon: History },
  { path: "/watchlist", label: "Watchlist", icon: Bell },
  { path: "/news-sentiment", label: "News", icon: Newspaper },
  { path: "/market-heatmap", label: "Heatmap", icon: Grid3X3 },
  { path: "/risk-analysis", label: "Risk", icon: ShieldCheck },
  { path: "/copilot", label: "Copilot", icon: Command },
  { path: "/assistant", label: "Assistant", icon: Bot },
];

const pageTitles: Record<string, string> = {
  "/dashboard": "AI Command Center",
  "/stock-analysis": "Stock Analysis",
  "/advanced-chart": "Advanced Chart",
  "/ai-prediction": "AI Prediction",
  "/explainable-ai": "Explainable AI",
  "/portfolio": "Portfolio Dashboard",
  "/optimizer": "Portfolio Optimizer",
  "/transactions": "Transaction History",
  "/watchlist": "Watchlist",
  "/news-sentiment": "News & Sentiment",
  "/market-heatmap": "Market Heatmap",
  "/risk-analysis": "Risk Analysis",
  "/copilot": "AI Investment Copilot",
  "/assistant": "AI Assistant",
};

function Shell() {
  const location = useLocation();
  const isApp = appNav.some((item) => location.pathname.startsWith(item.path));
  const isAuth = ["/login", "/register", "/forgot-password"].includes(location.pathname);

  return (
    <div className="min-h-screen overflow-x-hidden">
      <AnimatedBackground />
      {!isApp && !isAuth && <MarketingNav />}
      {isAuth && <AuthNav />}
      <AnimatePresence mode="wait">
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 16, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          exit={{ opacity: 0, y: -10, filter: "blur(8px)" }}
          transition={{ duration: 0.34, ease: "easeOut" }}
        >
          <Routes location={location}>
            <Route path="/" element={<LandingPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/features" element={<FeaturesPage />} />
            <Route path="/pricing" element={<PricingPage />} />
            <Route path="/contact" element={<ContactPage />} />
            <Route path="/login" element={<AuthPage mode="login" />} />
            <Route path="/register" element={<AuthPage mode="register" />} />
            <Route path="/forgot-password" element={<AuthPage mode="forgot" />} />
            <Route element={<AppLayout />}>
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/stock-analysis" element={<StockAnalysisPage />} />
              <Route path="/advanced-chart" element={<AdvancedChartPage />} />
              <Route path="/ai-prediction" element={<PredictionPage />} />
              <Route path="/explainable-ai" element={<ExplainableAiPage />} />
              <Route path="/portfolio" element={<PortfolioPage />} />
              <Route path="/optimizer" element={<OptimizerPage />} />
              <Route path="/transactions" element={<TransactionsPage />} />
              <Route path="/watchlist" element={<WatchlistPage />} />
              <Route path="/news-sentiment" element={<NewsPage />} />
              <Route path="/market-heatmap" element={<HeatmapPage />} />
              <Route path="/risk-analysis" element={<RiskPage />} />
              <Route path="/copilot" element={<CopilotPage />} />
              <Route path="/assistant" element={<AssistantPage />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

export default function App() {
  return <Shell />;
}

function AnimatedBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10">
      <div className="absolute inset-0 bg-[#050816]" />
      <div className="absolute inset-0 bg-grid opacity-[0.18]" />
      <motion.div
        className="absolute left-[-10%] top-[-20%] h-[34rem] w-[34rem] rounded-full bg-blue-500/20 blur-3xl"
        animate={{ x: [0, 60, 20], y: [0, 30, 70], scale: [1, 1.12, 0.98] }}
        transition={{ duration: 13, repeat: Infinity, repeatType: "mirror" }}
      />
      <motion.div
        className="absolute right-[-12%] top-[12%] h-[30rem] w-[30rem] rounded-full bg-cyan-500/14 blur-3xl"
        animate={{ x: [0, -80, -30], y: [0, 40, -10], scale: [1, 0.94, 1.1] }}
        transition={{ duration: 16, repeat: Infinity, repeatType: "mirror" }}
      />
      <motion.div
        className="absolute bottom-[-24%] left-[35%] h-[34rem] w-[34rem] rounded-full bg-violet-500/16 blur-3xl"
        animate={{ x: [0, 30, -50], y: [0, -50, -20], scale: [1, 1.1, 1] }}
        transition={{ duration: 18, repeat: Infinity, repeatType: "mirror" }}
      />
    </div>
  );
}

function Logo() {
  return (
    <Link to="/" className="flex items-center gap-3">
      <span className="grid h-10 w-10 place-items-center rounded-lg border border-blue-300/30 bg-blue-500/20 shadow-glow">
        <RadarIcon size={20} />
      </span>
      <span>
        <span className="block text-sm font-black tracking-wide">StockVision AI</span>
        <span className="block text-[11px] text-slate-400">Neural market intelligence</span>
      </span>
    </Link>
  );
}

function MarketingNav() {
  const [open, setOpen] = useState(false);
  return (
    <header className="fixed left-1/2 top-4 z-50 w-[min(1180px,calc(100%-24px))] -translate-x-1/2 rounded-lg border border-white/10 bg-[#07101f]/78 px-3 py-2 shadow-2xl backdrop-blur-2xl">
      <div className="flex items-center justify-between">
        <Logo />
        <nav className="hidden items-center gap-1 md:flex">
          {marketingPages.map((item) => (
            <NavLink key={item.path} to={item.path} className={({ isActive }) => cn("rounded-md px-3 py-2 text-sm text-slate-400 transition hover:bg-white/8 hover:text-white", isActive && "bg-white/10 text-white")}>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="hidden items-center gap-2 sm:flex">
          <Button variant="ghost" asChild>
            <Link to="/login">Login</Link>
          </Button>
          <Button asChild>
            <Link to="/dashboard">Open Platform</Link>
          </Button>
        </div>
        <button className="grid h-10 w-10 place-items-center rounded-md bg-white/5 md:hidden" onClick={() => setOpen((value) => !value)} aria-label="Toggle mobile navigation">
          {open ? <X size={18} /> : <Menu size={18} />}
        </button>
      </div>
      {open && (
        <div className="mt-3 grid gap-1 border-t border-white/10 pt-3 md:hidden">
          {[...marketingPages, { path: "/login", label: "Login" }, { path: "/dashboard", label: "Platform" }].map((item) => (
            <Link key={item.path} to={item.path} onClick={() => setOpen(false)} className="rounded-md px-3 py-2 text-sm text-slate-300 hover:bg-white/10">
              {item.label}
            </Link>
          ))}
        </div>
      )}
    </header>
  );
}

function AuthNav() {
  return (
    <header className="fixed left-1/2 top-4 z-50 flex w-[min(1120px,calc(100%-24px))] -translate-x-1/2 items-center justify-between rounded-lg border border-white/10 bg-[#07101f]/78 px-3 py-2 backdrop-blur-2xl">
      <Logo />
      <Button variant="ghost" asChild>
        <Link to="/">Back Home</Link>
      </Button>
    </header>
  );
}

function LandingPage() {
  return (
    <main className="pt-24">
      <section className="mx-auto grid min-h-[calc(100vh-96px)] w-[min(1180px,calc(100%-28px))] items-center gap-8 pb-10 lg:grid-cols-[1fr_0.82fr]">
        <div className="relative z-10">
          <motion.p initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="inline-flex rounded-md border border-cyan-300/25 bg-cyan-400/10 px-3 py-1 text-sm text-cyan-100">
            Institutional intelligence for modern investors
          </motion.p>
          <motion.h1 initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }} className="mt-5 max-w-4xl text-5xl font-black leading-[0.98] tracking-normal text-white md:text-7xl">
            StockVision AI
          </motion.h1>
          <motion.p initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.16 }} className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
            A futuristic stock intelligence platform for AI predictions, explainability, risk, portfolios, sentiment, live charting, and a financial copilot.
          </motion.p>
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.24 }} className="mt-8 flex flex-wrap gap-3">
            <Button asChild className="gap-2">
              <Link to="/dashboard">Launch Dashboard <ArrowRight size={16} /></Link>
            </Button>
            <Button variant="ghost" asChild>
              <Link to="/features">Explore Features</Link>
            </Button>
          </motion.div>
          <div className="mt-10 grid max-w-2xl grid-cols-3 gap-3">
            <Stat value="$2.4M" label="Tracked AUM" />
            <Stat value="92%" label="Top AI Confidence" />
            <Stat value="18ms" label="Signal Refresh" />
          </div>
        </div>
        <HeroTerminal />
      </section>
      <section className="mx-auto grid w-[min(1180px,calc(100%-28px))] gap-4 py-10 md:grid-cols-3">
        {heroFeatures.map(({ title, copy, icon: Icon }, index) => (
          <MotionCard key={title} delay={index * 0.08}>
            <Icon className="text-cyan-300" />
            <h3 className="mt-5 text-xl font-bold">{title}</h3>
            <p className="mt-3 text-sm leading-6 text-slate-400">{copy}</p>
          </MotionCard>
        ))}
      </section>
    </main>
  );
}

function HeroTerminal() {
  return (
    <motion.div initial={{ opacity: 0, scale: 0.96, y: 30 }} animate={{ opacity: 1, scale: 1, y: 0 }} transition={{ duration: 0.6 }} className="relative">
      <div className="absolute inset-0 rounded-lg bg-blue-500/20 blur-2xl" />
      <Card className="relative overflow-hidden p-0">
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
          <div className="flex gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-400" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-300" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
          </div>
          <span className="text-xs text-slate-500">SVAI / neural-market-os</span>
        </div>
        <div className="h-[460px] p-5">
          <div className="grid grid-cols-3 gap-3">
            {stocks.slice(0, 3).map((stock) => <Ticker key={stock.symbol} stock={stock} />)}
          </div>
          <div className="mt-5 h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="heroArea" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.65} />
                    <stop offset="100%" stopColor="#8B5CF6" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(148,163,184,.12)" vertical={false} />
                <XAxis dataKey="day" hide />
                <YAxis hide />
                <Tooltip content={<ChartTooltip />} />
                <Area type="monotone" dataKey="price" stroke="#60A5FA" strokeWidth={3} fill="url(#heroArea)" />
                <Line type="monotone" dataKey="ai" stroke="#22D3EE" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-5 grid gap-3">
            {["BUY NVDA: upside momentum +4.8%", "HOLD AAPL: services margin expanding", "REDUCE TSLA: volatility regime elevated"].map((item) => (
              <div key={item} className="flex items-center justify-between rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-sm">
                <span>{item}</span>
                <Sparkles size={15} className="text-cyan-300" />
              </div>
            ))}
          </div>
        </div>
      </Card>
    </motion.div>
  );
}

function AppLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const title = pageTitles[location.pathname] ?? "Workspace";
  const crumbs = title.split(" ");

  return (
    <div className="min-h-screen lg:pl-[var(--sidebar-width)]" style={{ "--sidebar-width": collapsed ? "92px" : "280px" } as React.CSSProperties}>
      <aside className={cn("fixed inset-y-0 left-0 z-50 hidden border-r border-white/10 bg-[#07101f]/82 p-3 backdrop-blur-2xl transition-all lg:block", collapsed ? "w-[92px]" : "w-[280px]")}>
        <div className="flex items-center justify-between">
          {!collapsed && <Logo />}
          {collapsed && <span className="grid h-10 w-10 place-items-center rounded-lg bg-blue-500/20"><RadarIcon size={19} /></span>}
          <button className="grid h-9 w-9 place-items-center rounded-md bg-white/5 text-slate-300" onClick={() => setCollapsed((value) => !value)} aria-label="Collapse sidebar">
            {collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}
          </button>
        </div>
        <nav className="mt-7 grid gap-1">
          {appNav.map((item) => <SidebarLink key={item.path} item={item} collapsed={collapsed} />)}
        </nav>
      </aside>

      <header className="sticky top-0 z-40 border-b border-white/10 bg-[#050816]/78 backdrop-blur-2xl">
        <div className="flex min-h-16 items-center justify-between gap-3 px-4 lg:px-7">
          <div className="flex items-center gap-3">
            <button className="grid h-10 w-10 place-items-center rounded-md bg-white/5 lg:hidden" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
              <Menu size={18} />
            </button>
            <div>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Link to="/" className="hover:text-white">Home</Link>
                <ChevronRight size={12} />
                <span>Platform</span>
                <ChevronRight size={12} />
                <span className="text-slate-300">{crumbs[0]}</span>
              </div>
              <h1 className="mt-1 text-lg font-bold">{title}</h1>
            </div>
          </div>
          <div className="hidden min-w-72 items-center gap-2 rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 md:flex">
            <Search size={16} className="text-slate-500" />
            <span className="text-sm text-slate-500">Search ticker, news, model, transaction...</span>
          </div>
          <div className="flex items-center gap-2">
            <IconButton icon={Bell} />
            <Button asChild className="hidden gap-2 sm:inline-flex">
              <Link to="/copilot"><Sparkles size={16} /> Ask AI</Link>
            </Button>
          </div>
        </div>
      </header>

      {mobileOpen && (
        <div className="fixed inset-0 z-[60] bg-black/60 lg:hidden" onClick={() => setMobileOpen(false)}>
          <aside className="h-full w-[min(310px,86vw)] border-r border-white/10 bg-[#07101f] p-3" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-center justify-between">
              <Logo />
              <button className="grid h-9 w-9 place-items-center rounded-md bg-white/5" onClick={() => setMobileOpen(false)} aria-label="Close navigation">
                <X size={17} />
              </button>
            </div>
            <nav className="mt-7 grid gap-1">
              {appNav.map((item) => <SidebarLink key={item.path} item={item} collapsed={false} onClick={() => setMobileOpen(false)} />)}
            </nav>
          </aside>
        </div>
      )}

      <main className="px-4 py-5 lg:px-7">
        <Routes>
          <Route path="*" element={null} />
        </Routes>
        <PageOutlet />
      </main>
    </div>
  );
}

function PageOutlet() {
  const location = useLocation();
  const Component = {
    "/dashboard": DashboardPage,
    "/stock-analysis": StockAnalysisPage,
    "/advanced-chart": AdvancedChartPage,
    "/ai-prediction": PredictionPage,
    "/explainable-ai": ExplainableAiPage,
    "/portfolio": PortfolioPage,
    "/optimizer": OptimizerPage,
    "/transactions": TransactionsPage,
    "/watchlist": WatchlistPage,
    "/news-sentiment": NewsPage,
    "/market-heatmap": HeatmapPage,
    "/risk-analysis": RiskPage,
    "/copilot": CopilotPage,
    "/assistant": AssistantPage,
  }[location.pathname] ?? DashboardPage;
  return <Component />;
}

function SidebarLink({ item, collapsed, onClick }: { item: (typeof appNav)[number]; collapsed: boolean; onClick?: () => void }) {
  const Icon = item.icon;
  return (
    <NavLink to={item.path} onClick={onClick} className={({ isActive }) => cn("group flex h-10 items-center gap-3 rounded-md px-3 text-sm text-slate-400 transition hover:bg-white/8 hover:text-white", isActive && "bg-blue-500/18 text-white shadow-glow")}>
      <Icon size={17} />
      {!collapsed && <span>{item.label}</span>}
    </NavLink>
  );
}

function DashboardPage() {
  const { widgets, moveWidget } = useAppStore();
  const widgetMap: Record<string, React.ReactNode> = {
    portfolio: <MetricCard title="Portfolio Value" value="$428,940" delta="+$12,420 today" icon={WalletCards} tone="emerald" />,
    profitLoss: <MetricCard title="Daily Profit" value="+$8,218" delta="+2.14% blended" icon={BadgeDollarSign} tone="emerald" />,
    confidence: <MetricCard title="AI Confidence" value="88/100" delta="NVDA leads signal set" icon={Brain} tone="blue" />,
    sentiment: <MetricCard title="Market Sentiment" value="Bullish" delta="0.72 composite" icon={Activity} tone="cyan" />,
    watchlist: <WatchlistMini />,
    news: <NewsMini />,
    prediction: <PredictionMini />,
    trending: <TrendingMini />,
  };

  return (
    <div className="grid gap-5">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {widgets.slice(0, 4).map((key, index) => (
          <motion.div key={key} layout drag dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }} whileDrag={{ scale: 1.03, zIndex: 20 }} onDragEnd={(_, info) => info.offset.x > 80 && index < widgets.length - 1 ? moveWidget(index, index + 1) : info.offset.x < -80 && index > 0 ? moveWidget(index, index - 1) : undefined}>
            {widgetMap[key]}
          </motion.div>
        ))}
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.35fr_0.65fr]">
        <MotionCard className="min-h-[440px]">
          <PanelHeader eyebrow="Portfolio chart" title="AI-adjusted equity curve" action="Zoom + pan enabled" />
          <div className="mt-5 h-[340px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData}>
                <CartesianGrid stroke="rgba(148,163,184,.11)" vertical={false} />
                <XAxis dataKey="day" tick={{ fill: "#64748b", fontSize: 11 }} />
                <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
                <Tooltip content={<ChartTooltip />} />
                <Area type="monotone" dataKey="price" fill="#3B82F626" stroke="#3B82F6" strokeWidth={3} />
                <Line type="monotone" dataKey="ai" stroke="#06B6D4" strokeWidth={2} dot={false} />
                <Bar dataKey="volume" fill="#8B5CF633" radius={[4, 4, 0, 0]} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </MotionCard>
        <div className="grid gap-5">
          {widgets.slice(4).map((key, index) => (
            <motion.div key={key} layout drag="y" dragConstraints={{ top: 0, bottom: 0 }} whileHover={{ y: -3 }} onDragEnd={(_, info) => info.offset.y > 40 && moveWidget(index + 4, Math.min(widgets.length - 1, index + 5))}>
              {widgetMap[key]}
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StockAnalysisPage() {
  const { selectedSymbol, setSymbol } = useAppStore();
  const [query, setQuery] = useState(selectedSymbol);
  return (
    <div className="grid gap-5 xl:grid-cols-[1.4fr_0.6fr]">
      <MotionCard>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <PanelHeader eyebrow="Research terminal" title={`${selectedSymbol} intelligence`} />
          <div className="flex gap-2">
            <Input value={query} onChange={(event) => setQuery(event.target.value.toUpperCase())} className="w-40" />
            <Button onClick={() => setSymbol(query)}>Analyze</Button>
          </div>
        </div>
        <AdvancedMarketChart />
      </MotionCard>
      <div className="grid gap-5">
        <MetricCard title="Live Price" value="$212.44" delta="+1.21% today" icon={CircleDollarSign} tone="emerald" />
        <MetricCard title="Model Consensus" value="Buy" delta="4 of 5 engines agree" icon={Brain} tone="blue" />
        <MotionCard>
          <PanelHeader eyebrow="Company DNA" title="Factor snapshot" />
          <FactorList />
        </MotionCard>
      </div>
    </div>
  );
}

function AdvancedChartPage() {
  return (
    <div className="grid gap-5">
      <MotionCard className="min-h-[650px]">
        <PanelHeader eyebrow="TradingView widget" title="Professional chart workspace" action="Live widget container" />
        <div className="mt-5 h-[560px] overflow-hidden rounded-lg border border-white/10 bg-[#0F172A]">
          <iframe title="TradingView Advanced Chart" className="h-full w-full" src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview&symbol=NASDAQ%3AAAPL&interval=D&hidesidetoolbar=0&symboledit=1&saveimage=0&toolbarbg=0F172A&studies=%5B%5D&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1" />
        </div>
      </MotionCard>
    </div>
  );
}

function PredictionPage() {
  return (
    <div className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
      <MotionCard>
        <PanelHeader eyebrow="AI prediction" title="Ensemble recommendation" />
        <div className="mt-8 text-center">
          <p className="gradient-text text-7xl font-black">BUY</p>
          <p className="mt-3 text-slate-400">Expected 30-day range: $224.10 - $238.70</p>
        </div>
        <div className="mt-8 grid grid-cols-3 gap-3">
          <Stat value="88%" label="Confidence" />
          <Stat value="+9.4%" label="Upside" />
          <Stat value="Low" label="Regime Risk" />
        </div>
      </MotionCard>
      <MotionCard>
        <PanelHeader eyebrow="Model comparison" title="Forecast engines" />
        <div className="mt-5 space-y-3">
          {["Gradient Boost", "Temporal CNN", "Sentiment LLM", "Macro Blend", "Momentum ARIMA"].map((model, index) => (
            <ModelRow key={model} model={model} score={92 - index * 5} price={229 - index * 1.7} />
          ))}
        </div>
      </MotionCard>
    </div>
  );
}

function ExplainableAiPage() {
  const data = [
    { factor: "Earnings Revision", impact: 92 },
    { factor: "Volume Shock", impact: 81 },
    { factor: "News Tone", impact: 74 },
    { factor: "RSI Reset", impact: 62 },
    { factor: "Macro Liquidity", impact: 55 },
  ];
  return (
    <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
      <MotionCard>
        <PanelHeader eyebrow="Explainability" title="Feature importance" />
        <div className="mt-6 h-[420px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical">
              <CartesianGrid stroke="rgba(148,163,184,.11)" horizontal={false} />
              <XAxis type="number" hide />
              <YAxis dataKey="factor" type="category" tick={{ fill: "#cbd5e1", fontSize: 12 }} width={130} />
              <Tooltip content={<ChartTooltip />} />
              <Bar dataKey="impact" fill="#8B5CF6" radius={[0, 7, 7, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </MotionCard>
      <MotionCard>
        <PanelHeader eyebrow="Why this matters" title="AI rationale" />
        <div className="mt-5 space-y-3 text-sm leading-6 text-slate-300">
          {[
            "Positive revisions and accelerating volume explain most of the bullish forecast.",
            "Sentiment is supportive but less decisive than fundamentals this session.",
            "Risk model flags a moderate drawdown window if broad tech beta weakens.",
          ].map((item) => <p key={item} className="rounded-md border border-white/10 bg-white/[0.04] p-4">{item}</p>)}
        </div>
      </MotionCard>
    </div>
  );
}

function PortfolioPage() {
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_0.85fr]">
      <MotionCard>
        <PanelHeader eyebrow="Portfolio dashboard" title="Allocation and performance" />
        <div className="mt-5 h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={holdings} dataKey="value" nameKey="name" innerRadius={76} outerRadius={116} paddingAngle={4}>
                {holdings.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
              </Pie>
              <Tooltip content={<ChartTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </MotionCard>
      <div className="grid gap-5">
        <MetricCard title="Total Value" value="$428,940" delta="+18.7% YTD" icon={BriefcaseBusiness} tone="emerald" />
        <HoldingsTable />
      </div>
    </div>
  );
}

function OptimizerPage() {
  return (
    <div className="grid gap-5 xl:grid-cols-[0.85fr_1.15fr]">
      <MotionCard>
        <PanelHeader eyebrow="Optimizer" title="Target constraints" />
        <div className="mt-5 space-y-4">
          {["Max position 22%", "Min cash 7%", "Reduce beta below 1.15", "Increase AI exposure"].map((item) => <ToggleRow key={item} label={item} />)}
          <Button className="w-full gap-2"><Sparkles size={16} /> Generate Rebalance</Button>
        </div>
      </MotionCard>
      <MotionCard>
        <PanelHeader eyebrow="Efficient frontier" title="Risk-adjusted allocations" />
        <div className="mt-5 h-[360px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData.slice(0, 38)}>
              <CartesianGrid stroke="rgba(148,163,184,.11)" />
              <XAxis dataKey="volume" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip content={<ChartTooltip />} />
              <Line type="monotone" dataKey="price" stroke="#10B981" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="ai" stroke="#3B82F6" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </MotionCard>
    </div>
  );
}

function TransactionsPage() {
  return <DataTable title="Transaction History" rows={["Bought 12 NVDA @ $131.20", "Sold 5 TSLA @ $188.40", "Dividend AAPL $18.30", "Bought 8 MSFT @ $421.10", "Rebalanced cash reserve +$4,200"]} />;
}

function WatchlistPage() {
  return (
    <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
      {stocks.map((stock, index) => (
        <MotionCard key={stock.symbol} delay={index * 0.05}>
          <Ticker stock={stock} />
          <div className="mt-5 h-28">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData.slice(index, index + 24)}>
                <Area dataKey="price" stroke={stock.change >= 0 ? "#10B981" : "#EF4444"} fill={stock.change >= 0 ? "#10B98122" : "#EF444422"} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </MotionCard>
      ))}
    </div>
  );
}

function NewsPage() {
  return (
    <div className="grid gap-5 xl:grid-cols-[0.75fr_1.25fr]">
      <MotionCard>
        <PanelHeader eyebrow="Sentiment" title="Market emotion radar" />
        <div className="mt-5 h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={[{ k: "AI", v: 92 }, { k: "Banks", v: 64 }, { k: "Energy", v: 45 }, { k: "Retail", v: 58 }, { k: "Crypto", v: 75 }]}>
              <PolarGrid stroke="rgba(148,163,184,.18)" />
              <PolarAngleAxis dataKey="k" tick={{ fill: "#cbd5e1", fontSize: 12 }} />
              <PolarRadiusAxis tick={false} axisLine={false} />
              <Radar dataKey="v" stroke="#06B6D4" fill="#06B6D4" fillOpacity={0.28} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </MotionCard>
      <DataTable title="AI-Classified News" rows={news} />
    </div>
  );
}

function HeatmapPage() {
  return (
    <MotionCard className="min-h-[650px]">
      <PanelHeader eyebrow="Market map" title="Sector and ticker performance" />
      <div className="mt-5 h-[540px]">
        <ResponsiveContainer width="100%" height="100%">
          <Treemap data={heatmap} dataKey="size" nameKey="name" stroke="#050816" fill="#10B981" content={<HeatmapTile />} />
        </ResponsiveContainer>
      </div>
    </MotionCard>
  );
}

function RiskPage() {
  return (
    <div className="grid gap-5 xl:grid-cols-4">
      <MetricCard title="Portfolio Beta" value="1.08" delta="Below target" icon={Gauge} tone="blue" />
      <MetricCard title="Sharpe Ratio" value="1.74" delta="+0.22 QoQ" icon={TrendingUp} tone="emerald" />
      <MetricCard title="Max Drawdown" value="-8.6%" delta="Stress case" icon={ShieldCheck} tone="red" />
      <MetricCard title="VaR 95%" value="$14.2K" delta="1 day horizon" icon={Landmark} tone="cyan" />
      <MotionCard className="xl:col-span-4">
        <PanelHeader eyebrow="Risk simulation" title="Scenario drawdown model" />
        <div className="mt-5 h-[360px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <CartesianGrid stroke="rgba(148,163,184,.11)" />
              <XAxis dataKey="day" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip content={<ChartTooltip />} />
              <Area dataKey="sentiment" stroke="#EF4444" fill="#EF444426" />
              <Area dataKey="volume" stroke="#8B5CF6" fill="#8B5CF626" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </MotionCard>
    </div>
  );
}

function CopilotPage() {
  return <AiChat title="AI Investment Copilot" prompt="Rebalance my portfolio for lower beta without losing AI upside." />;
}

function AssistantPage() {
  return <AiChat title="AI Assistant" prompt="Should I buy Tesla this week?" compact />;
}

function AboutPage() {
  return <MarketingPage title="Built for signal, speed, and trust" copy="StockVision AI turns fragmented market data into a calm operating system for investors. The interface is designed for repeat daily use: dense where it matters, cinematic where it inspires confidence." icon={Compass} />;
}

function FeaturesPage() {
  return (
    <main className="mx-auto w-[min(1180px,calc(100%-28px))] pt-28">
      <SectionTitle eyebrow="Features" title="A complete AI market workstation" />
      <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {appNav.slice(1).map((item, index) => {
          const Icon = item.icon;
          return (
            <MotionCard key={item.path} delay={index * 0.03}>
              <Icon className="text-blue-300" />
              <h3 className="mt-4 text-xl font-bold">{item.label}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">Production-grade workflows, dummy data, responsive layout, motion, and reusable UI architecture.</p>
            </MotionCard>
          );
        })}
      </div>
    </main>
  );
}

function PricingPage() {
  return (
    <main className="mx-auto w-[min(1180px,calc(100%-28px))] pt-28">
      <SectionTitle eyebrow="Pricing" title="Simple plans for every investing desk" />
      <div className="mt-8 grid gap-4 lg:grid-cols-3">
        {["Starter", "Pro", "Institutional"].map((plan, index) => (
          <MotionCard key={plan} delay={index * 0.08} className={index === 1 ? "border-blue-300/40 shadow-glow" : ""}>
            <h3 className="text-2xl font-black">{plan}</h3>
            <p className="mt-3 text-4xl font-black">{index === 0 ? "$0" : index === 1 ? "$29" : "Custom"}</p>
            <div className="mt-6 space-y-3 text-sm text-slate-300">
              {["AI dashboard", "Watchlist and alerts", "Portfolio analytics", "Copilot workflows"].map((item) => <p key={item}><Check className="mr-2 inline text-emerald-300" size={16} />{item}</p>)}
            </div>
            <Button className="mt-7 w-full" asChild><Link to="/register">Start Now</Link></Button>
          </MotionCard>
        ))}
      </div>
    </main>
  );
}

function ContactPage() {
  return (
    <main className="mx-auto grid min-h-screen w-[min(1180px,calc(100%-28px))] items-center gap-8 pt-24 lg:grid-cols-[0.9fr_1.1fr]">
      <SectionTitle eyebrow="Contact" title="Bring StockVision AI to your investing workflow" />
      <MotionCard>
        <div className="grid gap-3">
          <Input placeholder="Name" />
          <Input placeholder="Email" />
          <textarea className="min-h-36 rounded-md border border-white/10 bg-white/[0.04] p-3 text-sm outline-none transition focus:border-blue-300/60" placeholder="What are you building?" />
          <Button>Send Message</Button>
        </div>
      </MotionCard>
    </main>
  );
}

function MarketingPage({ title, copy, icon: Icon }: { title: string; copy: string; icon: React.ElementType }) {
  return (
    <main className="mx-auto grid min-h-screen w-[min(1180px,calc(100%-28px))] items-center gap-8 pt-24 lg:grid-cols-[0.9fr_1.1fr]">
      <div>
        <SectionTitle eyebrow="StockVision AI" title={title} />
        <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-300">{copy}</p>
      </div>
      <MotionCard className="min-h-96">
        <Icon size={48} className="text-cyan-300" />
        <div className="mt-8 grid grid-cols-2 gap-3">
          <Stat value="22" label="Product screens" />
          <Stat value="14" label="App modules" />
          <Stat value="100%" label="Dark mode" />
          <Stat value="5" label="AI lenses" />
        </div>
      </MotionCard>
    </main>
  );
}

function AuthPage({ mode }: { mode: "login" | "register" | "forgot" }) {
  const copy = mode === "login" ? "Welcome back to your intelligence desk." : mode === "register" ? "Create your market command center." : "Reset access to your workspace.";
  return (
    <main className="mx-auto grid min-h-screen w-[min(1080px,calc(100%-28px))] items-center gap-8 pt-24 lg:grid-cols-[1fr_430px]">
      <div>
        <p className="text-sm uppercase text-blue-300">Secure Access</p>
        <h1 className="mt-3 text-5xl font-black leading-tight">{copy}</h1>
        <p className="mt-4 max-w-xl text-slate-400">A polished Firebase/JWT-ready auth surface with dedicated login, register, and forgot password routes.</p>
      </div>
      <MotionCard>
        <div className="flex gap-2">
          <Button variant={mode === "login" ? "primary" : "ghost"} asChild><Link to="/login">Login</Link></Button>
          <Button variant={mode === "register" ? "primary" : "ghost"} asChild><Link to="/register">Register</Link></Button>
          <Button variant={mode === "forgot" ? "primary" : "ghost"} asChild><Link to="/forgot-password">Reset</Link></Button>
        </div>
        <div className="mt-6 grid gap-3">
          {mode === "register" && <Input placeholder="Full name" />}
          <Input placeholder="Email" />
          {mode !== "forgot" && <Input type="password" placeholder="Password" />}
          <Button className="gap-2">{mode === "forgot" ? <Mail size={16} /> : mode === "register" ? <UserPlus size={16} /> : <Lock size={16} />}{mode === "forgot" ? "Send Reset Link" : mode === "register" ? "Create Account" : "Login"}</Button>
        </div>
      </MotionCard>
    </main>
  );
}

function AdvancedMarketChart() {
  return (
    <div className="mt-5">
      <div className="mb-4 flex flex-wrap gap-2">
        {["1D", "5D", "1M", "6M", "1Y", "MAX", "RSI", "MACD", "VWAP"].map((item) => <button key={item} className="rounded-md border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-slate-300 transition hover:bg-white/10">{item}</button>)}
      </div>
      <div className="h-[420px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={candles}>
            <CartesianGrid stroke="rgba(148,163,184,.11)" vertical={false} />
            <XAxis dataKey="day" tick={{ fill: "#64748b", fontSize: 11 }} />
            <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="high" fill="#1E293B" radius={[5, 5, 0, 0]} />
            <Bar dataKey="close" fill="#10B981" radius={[5, 5, 0, 0]} />
            <Line type="monotone" dataKey="open" stroke="#06B6D4" dot={false} strokeWidth={2} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function AiChat({ title, prompt, compact }: { title: string; prompt: string; compact?: boolean }) {
  const [message, setMessage] = useState(prompt);
  return (
    <div className={cn("grid gap-5", compact ? "xl:grid-cols-[1fr_0.8fr]" : "xl:grid-cols-[1.25fr_0.75fr]")}>
      <MotionCard className="min-h-[620px]">
        <PanelHeader eyebrow="Conversational intelligence" title={title} />
        <div className="mt-6 space-y-4">
          <ChatBubble role="ai">I have analyzed current momentum, sentiment, portfolio exposure, and risk. Ask me to compare tickers, explain a signal, or produce a rebalance.</ChatBubble>
          <ChatBubble role="user">{message}</ChatBubble>
          <ChatBubble role="ai">My current view is constructive but risk-aware: keep AI exposure, cap single-name concentration, and use a staged entry if volatility expands.</ChatBubble>
        </div>
        <div className="mt-6 flex gap-2">
          <Input value={message} onChange={(event) => setMessage(event.target.value)} className="flex-1" />
          <Button className="gap-2"><MessageSquareText size={16} /> Send</Button>
        </div>
      </MotionCard>
      <div className="grid gap-5">
        <MetricCard title="Context Sources" value="8" delta="news, chart, risk, filings" icon={ClipboardList} tone="blue" />
        <DataTable title="Suggested Prompts" rows={["Compare NVDA and AMD for 30 days", "Explain my portfolio drawdown risk", "Find undervalued AI infrastructure stocks"]} />
      </div>
    </div>
  );
}

function DataTable({ title, rows }: { title: string; rows: string[] }) {
  return (
    <MotionCard>
      <PanelHeader eyebrow="Records" title={title} />
      <div className="mt-5 space-y-3">
        {rows.map((row, index) => (
          <div key={row} className="flex items-center justify-between rounded-md border border-white/10 bg-white/[0.04] p-4 text-sm">
            <span className="text-slate-300">{row}</span>
            <span className="text-xs text-slate-500">#{String(index + 1).padStart(3, "0")}</span>
          </div>
        ))}
      </div>
    </MotionCard>
  );
}

function WatchlistMini() {
  return (
    <MotionCard>
      <PanelHeader eyebrow="Watchlist" title="Live targets" />
      <div className="mt-4 space-y-3">{stocks.slice(0, 4).map((stock) => <Ticker key={stock.symbol} stock={stock} compact />)}</div>
    </MotionCard>
  );
}

function NewsMini() {
  return <DataTable title="Recent News" rows={news.slice(0, 3)} />;
}

function PredictionMini() {
  return (
    <MotionCard>
      <PanelHeader eyebrow="Trending stocks" title="AI signal board" />
      <div className="mt-4 space-y-3">{stocks.slice(0, 3).map((stock) => <ModelRow key={stock.symbol} model={stock.symbol} score={stock.confidence} price={stock.price} />)}</div>
    </MotionCard>
  );
}

function TrendingMini() {
  return (
    <MotionCard>
      <PanelHeader eyebrow="Market sentiment" title="Composite index" />
      <div className="mt-4 h-28">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData.slice(0, 28)}>
            <Area dataKey="sentiment" stroke="#06B6D4" fill="#06B6D422" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </MotionCard>
  );
}

function HoldingsTable() {
  return <DataTable title="Holdings" rows={stocks.slice(0, 4).map((stock) => `${stock.symbol} - ${stock.sector} - ${currency(stock.price)} - ${percent(stock.change)}`)} />;
}

function FactorList() {
  return (
    <div className="mt-4 space-y-3">
      {["Revenue acceleration", "Options skew", "Relative strength", "Analyst revisions"].map((item, index) => (
        <div key={item}>
          <div className="flex justify-between text-sm"><span>{item}</span><span>{86 - index * 9}%</span></div>
          <div className="mt-2 h-2 rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400" style={{ width: `${86 - index * 9}%` }} /></div>
        </div>
      ))}
    </div>
  );
}

function ToggleRow({ label }: { label: string }) {
  const [on, setOn] = useState(true);
  return (
    <button onClick={() => setOn((value) => !value)} className="flex w-full items-center justify-between rounded-md border border-white/10 bg-white/[0.04] p-3 text-left text-sm">
      <span>{label}</span>
      <span className={cn("h-6 w-11 rounded-full p-1 transition", on ? "bg-blue-500" : "bg-slate-700")}>
        <span className={cn("block h-4 w-4 rounded-full bg-white transition", on && "translate-x-5")} />
      </span>
    </button>
  );
}

function MetricCard({ title, value, delta, icon: Icon, tone }: { title: string; value: string; delta: string; icon: React.ElementType; tone: "blue" | "emerald" | "cyan" | "red" }) {
  const toneClass = { blue: "text-blue-300 bg-blue-500/14", emerald: "text-emerald-300 bg-emerald-500/14", cyan: "text-cyan-300 bg-cyan-500/14", red: "text-red-300 bg-red-500/14" }[tone];
  return (
    <MotionCard className="group">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-400">{title}</p>
          <p className="mt-3 text-3xl font-black tracking-normal">{value}</p>
          <p className="mt-2 text-sm text-slate-500">{delta}</p>
        </div>
        <span className={cn("grid h-11 w-11 place-items-center rounded-lg transition group-hover:scale-110", toneClass)}><Icon size={20} /></span>
      </div>
    </MotionCard>
  );
}

function Ticker({ stock, compact }: { stock: (typeof stocks)[number]; compact?: boolean }) {
  return (
    <div className={cn("rounded-md border border-white/10 bg-white/[0.04] p-3", compact && "p-2.5")}>
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="font-bold">{stock.symbol}</p>
          {!compact && <p className="text-xs text-slate-500">{stock.name}</p>}
        </div>
        <div className="text-right">
          <p className="font-semibold">{currency(stock.price)}</p>
          <p className={cn("text-xs", stock.change >= 0 ? "text-emerald-300" : "text-red-300")}>{percent(stock.change)}</p>
        </div>
      </div>
    </div>
  );
}

function ModelRow({ model, score, price }: { model: string; score: number; price: number }) {
  return (
    <div className="rounded-md border border-white/10 bg-white/[0.04] p-3">
      <div className="flex justify-between text-sm"><span>{model}</span><span>{currency(price)} / {score}%</span></div>
      <div className="mt-2 h-2 rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-blue-400" style={{ width: `${score}%` }} /></div>
    </div>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-white/[0.04] p-4">
      <AnimatedNumber value={value} />
      <p className="mt-1 text-xs text-slate-500">{label}</p>
    </div>
  );
}

function AnimatedNumber({ value }: { value: string }) {
  return <motion.p initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} className="gradient-text text-2xl font-black">{value}</motion.p>;
}

function PanelHeader({ eyebrow, title, action }: { eyebrow: string; title: string; action?: string }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-blue-300">{eyebrow}</p>
        <h2 className="mt-1 text-xl font-bold">{title}</h2>
      </div>
      {action && <span className="rounded-md border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-slate-400">{action}</span>}
    </div>
  );
}

function SectionTitle({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <div>
      <p className="text-sm uppercase text-cyan-300">{eyebrow}</p>
      <h1 className="mt-3 max-w-3xl text-5xl font-black leading-tight">{title}</h1>
    </div>
  );
}

function MotionCard({ children, className, delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-40px" }} transition={{ delay, duration: 0.4 }} whileHover={{ y: -4, rotateX: 1, rotateY: -1 }}>
      <Card className={cn("h-full transition duration-300 hover:border-blue-300/35 hover:shadow-glow", className)}>{children}</Card>
    </motion.div>
  );
}

function IconButton({ icon: Icon }: { icon: React.ElementType }) {
  return <button className="grid h-10 w-10 place-items-center rounded-md border border-white/10 bg-white/[0.04] text-slate-300 transition hover:bg-white/10" aria-label="Toolbar action"><Icon size={17} /></button>;
}

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-white/10 bg-[#07101f]/95 p-3 text-xs shadow-2xl backdrop-blur-xl">
      <p className="mb-1 text-slate-400">{label}</p>
      {payload.map((item: any) => <p key={item.dataKey} style={{ color: item.color }}>{item.name ?? item.dataKey}: {typeof item.value === "number" ? item.value.toFixed(2) : item.value}</p>)}
    </div>
  );
}

function HeatmapTile(props: any) {
  const { x, y, width, height, name, change } = props;
  const positive = change >= 0;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} rx={8} fill={positive ? "#10B981" : "#EF4444"} fillOpacity={0.2 + Math.min(Math.abs(change) / 8, 0.6)} stroke="rgba(255,255,255,.14)" />
      {width > 55 && height > 38 && (
        <>
          <text x={x + 12} y={y + 24} fill="#fff" fontSize={13} fontWeight={700}>{name}</text>
          <text x={x + 12} y={y + 44} fill={positive ? "#86efac" : "#fca5a5"} fontSize={12}>{percent(change)}</text>
        </>
      )}
    </g>
  );
}

function ChatBubble({ role, children }: { role: "ai" | "user"; children: React.ReactNode }) {
  return (
    <div className={cn("max-w-[82%] rounded-lg border p-4 text-sm leading-6", role === "ai" ? "border-blue-300/20 bg-blue-500/10 text-slate-200" : "ml-auto border-emerald-300/20 bg-emerald-500/10 text-emerald-50")}>
      {children}
    </div>
  );
}
