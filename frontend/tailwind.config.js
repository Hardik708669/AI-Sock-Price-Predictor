/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        canvas: 'var(--color-canvas)',
        bento: 'var(--color-bento)',
        'text-primary': 'var(--color-text-primary)',
        'text-secondary': 'var(--color-text-secondary)',
        accent: 'var(--color-accent)',
        pill: 'var(--color-pill)',
        bullish: 'var(--color-bullish)',
        bearish: 'var(--color-bearish)',
      },
      fontFamily: {
        sans: ['"DM Sans"', 'system-ui', 'sans-serif'],
        display: ['"Syne"', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        luxury: '0 8px 30px rgb(0 0 0 / 0.04)',
      },
      animation: {
        'soft-pulse': 'soft-pulse 2.5s ease-in-out infinite',
        'spin-slow': 'spin 3s linear infinite',
      },
      keyframes: {
        'soft-pulse': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(222, 93, 0, 0.4)' },
          '50%': { boxShadow: '0 0 0 12px rgba(222, 93, 0, 0)' },
        },
      },
    },
  },
  plugins: [],
}
