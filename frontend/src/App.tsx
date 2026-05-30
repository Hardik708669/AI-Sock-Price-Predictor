import { lazy, Suspense, useState } from "react";
import { Layout } from "./components/Layout";

const AuthPage = lazy(() => import("./components/AuthPage").then((module) => ({ default: module.AuthPage })));
const AssistantPage = lazy(() => import("./components/AssistantPage").then((module) => ({ default: module.AssistantPage })));
const Dashboard = lazy(() => import("./components/Dashboard").then((module) => ({ default: module.Dashboard })));
const LandingPage = lazy(() => import("./components/LandingPage").then((module) => ({ default: module.LandingPage })));
const PortfolioPage = lazy(() => import("./components/PortfolioPage").then((module) => ({ default: module.PortfolioPage })));
const StockAnalysis = lazy(() => import("./components/StockAnalysis").then((module) => ({ default: module.StockAnalysis })));

type View = "landing" | "auth" | "dashboard" | "analysis" | "portfolio" | "assistant";

export default function App() {
  const [view, setView] = useState<View>("landing");

  return (
    <Layout view={view} setView={setView}>
      <Suspense fallback={<div className="mx-auto w-[min(1180px,calc(100%-28px))] py-24 text-slate-400">Loading StockVision AI...</div>}>
        {view === "landing" && <LandingPage openDashboard={() => setView("dashboard")} />}
        {view === "auth" && <AuthPage />}
        {view === "dashboard" && <Dashboard />}
        {view === "analysis" && <StockAnalysis />}
        {view === "portfolio" && <PortfolioPage />}
        {view === "assistant" && <AssistantPage />}
      </Suspense>
    </Layout>
  );
}
