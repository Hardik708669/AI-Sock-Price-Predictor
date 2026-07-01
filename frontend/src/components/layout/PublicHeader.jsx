import { Link } from 'react-router-dom'
import ThemeToggle from '../ui/ThemeToggle'

export default function PublicHeader() {
  return (
    <header className="fixed left-0 right-0 top-0 z-50 px-6 py-5 transition-all duration-500">
      <div className="mx-auto flex max-w-7xl items-center justify-between">
        <Link
          to="/"
          className="font-display text-xl font-bold tracking-tight text-text-primary transition-colors duration-500"
        >
          Trade Vision
        </Link>

        <div className="flex items-center gap-3">
          <ThemeToggle />
          <Link
            to="/login"
            className="rounded-full px-5 py-2.5 text-sm font-medium text-text-secondary transition-all hover:text-text-primary active:scale-95"
          >
            Log In
          </Link>
          <Link
            to="/signup"
            className="rounded-full bg-accent px-5 py-2.5 text-sm font-semibold text-white transition-all hover:brightness-110 active:scale-95"
          >
            Sign Up
          </Link>
        </div>
      </div>
    </header>
  )
}
