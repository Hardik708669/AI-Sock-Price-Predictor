import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  LineChart,
  LogOut,
  Shield,
  TrendingUp,
  X,
} from 'lucide-react'
import { useAuth } from '../../context/AuthContext'
import ThemeToggle from '../ui/ThemeToggle'

const navItems = [
  { to: '/app/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/app/market-mood', label: 'Market Mood', icon: TrendingUp },
  { to: '/app/backtester', label: 'Backtester', icon: LineChart },
  { to: '/app/vault', label: 'Vault Portfolio', icon: Shield },
]

export default function AppSidebar({ mobileOpen, onClose }) {
  const { user, logout } = useAuth()

  const sidebarContent = (
    <>
      <div className="mb-8 px-2">
        <p className="font-display text-lg font-bold text-text-primary">Trade Vision</p>
        <p className="text-xs text-text-secondary">AI Trading Terminal</p>
      </div>

      <nav className="flex-1 space-y-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            onClick={onClose}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium transition-all active:scale-[0.98] ${
                isActive
                  ? 'bg-accent/10 text-accent'
                  : 'text-text-secondary hover:bg-pill/40 hover:text-text-primary'
              }`
            }
          >
            <Icon className="h-5 w-5" strokeWidth={1.5} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto space-y-4 border-t border-pill/30 pt-4">
        <div className="flex items-center justify-between px-2">
          <span className="text-xs text-text-secondary">Theme</span>
          <ThemeToggle />
        </div>
        <div className="rounded-2xl bg-pill/30 px-4 py-3">
          <p className="text-sm font-medium text-text-primary">{user?.name}</p>
          <p className="truncate text-xs text-text-secondary">{user?.email}</p>
        </div>
        <button
          onClick={logout}
          className="flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium text-text-secondary transition-all hover:bg-bearish/10 hover:text-bearish active:scale-[0.98]"
        >
          <LogOut className="h-5 w-5" strokeWidth={1.5} />
          Log Out
        </button>
      </div>
    </>
  )

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-pill/20 bg-bento p-6 lg:flex">
        {sidebarContent}
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
          <aside className="absolute left-0 top-0 flex h-full w-72 flex-col bg-bento p-6 shadow-2xl">
            <button
              onClick={onClose}
              className="mb-4 ml-auto flex h-9 w-9 items-center justify-center rounded-full bg-pill/40 active:scale-95"
            >
              <X className="h-4 w-4" strokeWidth={1.5} />
            </button>
            {sidebarContent}
          </aside>
        </div>
      )}
    </>
  )
}
