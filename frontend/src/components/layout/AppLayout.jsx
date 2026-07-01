import { useState } from 'react'
import { Menu } from 'lucide-react'
import { Outlet, useLocation } from 'react-router-dom'
import AppSidebar from './AppSidebar'

const pageTitles = {
  '/app/dashboard': 'Prediction Dashboard',
  '/app/market-mood': 'Market Mood Radar',
  '/app/backtester': 'Financial Time Machine',
  '/app/vault': 'Safety Guardrail Portfolio',
}

export default function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const { pathname } = useLocation()
  const title = pageTitles[pathname] || 'Trade Vision'

  return (
    <div className="flex min-h-screen bg-canvas transition-all duration-500 ease-in-out">
      <AppSidebar mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-40 flex items-center gap-4 border-b border-pill/20 bg-canvas/80 px-6 py-4 backdrop-blur-xl">
          <button
            onClick={() => setMobileOpen(true)}
            className="flex h-10 w-10 items-center justify-center rounded-xl bg-bento light-shadow lg:hidden active:scale-95"
          >
            <Menu className="h-5 w-5 text-text-primary" strokeWidth={1.5} />
          </button>
          <h1 className="font-display text-xl font-bold text-text-primary">{title}</h1>
        </header>

        <main className="flex-1 overflow-auto px-6 py-8 transition-all duration-500">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
