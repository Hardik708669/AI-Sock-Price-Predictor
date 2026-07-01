import { Moon, Sun } from 'lucide-react'
import { useTheme } from '../../context/ThemeContext'

export default function ThemeToggle() {
  const { isDark, toggleTheme } = useTheme()

  return (
    <button
      onClick={toggleTheme}
      aria-label="Toggle theme"
      className="relative flex h-10 w-[4.5rem] items-center rounded-full bg-pill p-1 transition-all duration-500 ease-in-out active:scale-95"
    >
      <span
        className={`absolute flex h-8 w-8 items-center justify-center rounded-full bg-bento light-shadow transition-all duration-500 ease-in-out ${
          isDark ? 'translate-x-[2.25rem] rotate-[360deg]' : 'translate-x-0 rotate-0'
        }`}
      >
        {isDark ? (
          <Moon className="h-4 w-4 text-accent" strokeWidth={1.5} />
        ) : (
          <Sun className="h-4 w-4 text-accent" strokeWidth={1.5} />
        )}
      </span>
    </button>
  )
}
