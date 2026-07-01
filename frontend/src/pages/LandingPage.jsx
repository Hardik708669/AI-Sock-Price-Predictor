import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Area,
  AreaChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Brain, Route, Radar, Shield } from 'lucide-react'
import { bentoFeatures, generatePrediction, generateStockHistory } from '../data/mockData'

const iconMap = { Brain, Route, Radar, Shield }

function GridBackground({ mouseX, mouseY }) {
  return (
    <div
      className="pointer-events-none fixed inset-0 overflow-hidden opacity-30 transition-all duration-500"
      style={{
        backgroundImage: `
          linear-gradient(var(--color-pill) 1px, transparent 1px),
          linear-gradient(90deg, var(--color-pill) 1px, transparent 1px)
        `,
        backgroundSize: '60px 60px',
        backgroundPosition: `${mouseX * 0.02}px ${mouseY * 0.02}px`,
      }}
    />
  )
}

function HeroChartTeaser() {
  const history = generateStockHistory(60, 248, 303)
  const forecast = generatePrediction(history[history.length - 1].price, 14, 505)
  const combined = [...history, ...forecast]
  const [sliderIdx, setSliderIdx] = useState(45)

  const historySlice = combined.filter((d) => d.type !== 'forecast')
  const visibleHistory = historySlice.slice(0, Math.min(sliderIdx + 1, historySlice.length))
  const forecastStart = Math.max(0, sliderIdx - historySlice.length + 1)
  const visibleForecast =
    sliderIdx >= historySlice.length - 1
      ? [historySlice[historySlice.length - 1], ...forecast.slice(0, forecastStart + 1)]
      : []
  const currentPoint = combined[sliderIdx]

  return (
    <div className="relative overflow-hidden rounded-3xl glass-surface p-6 light-shadow transition-all duration-500">
      <span className="pointer-events-none absolute inset-0 flex items-center justify-center font-display text-[8rem] font-bold text-text-primary/[0.03]">
        TSLA
      </span>

      <div className="relative z-10">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="text-xs font-medium uppercase tracking-widest text-text-secondary">Live Preview</p>
            <p className="font-display text-2xl font-bold text-text-primary">TSLA · ${currentPoint?.price?.toFixed(2)}</p>
          </div>
          <span className="rounded-full bg-accent/10 px-3 py-1 text-xs font-semibold text-accent">
            AI Forecast Active
          </span>
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={visibleHistory} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="heroGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#DE5D00" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#DE5D00" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="date" hide />
            <YAxis domain={['auto', 'auto']} hide />
            <Tooltip
              content={({ active, payload }) =>
                active && payload?.[0] ? (
                  <div className="rounded-xl bg-bento px-3 py-2 text-xs light-shadow">
                    <p className="font-semibold text-text-primary">${payload[0].value}</p>
                    <p className="text-text-secondary">{payload[0].payload.date}</p>
                  </div>
                ) : null
              }
            />
            <Area
              type="monotone"
              dataKey="price"
              stroke="#DE5D00"
              strokeWidth={2}
              fill="url(#heroGrad)"
              dot={false}
            />
            {visibleForecast.length > 1 && (
              <Area
                type="monotone"
                dataKey="price"
                data={visibleForecast}
                stroke="#DE5D00"
                strokeWidth={2}
                fill="none"
                dot={false}
                strokeDasharray="6 4"
              />
            )}
            {currentPoint && (
              <ReferenceDot
                x={currentPoint.date}
                y={currentPoint.price}
                r={6}
                fill="#DE5D00"
                stroke="#fff"
                strokeWidth={2}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>

        <div className="mt-4">
          <input
            type="range"
            min={10}
            max={combined.length - 1}
            value={sliderIdx}
            onChange={(e) => setSliderIdx(Number(e.target.value))}
            className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-pill accent-accent"
          />
          <div className="mt-2 flex justify-between text-xs text-text-secondary">
            <span>Historic</span>
            <span className="text-accent">Forecast →</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function LandingPage() {
  const [mouse, setMouse] = useState({ x: 0, y: 0 })
  const containerRef = useRef(null)

  useEffect(() => {
    const handler = (e) => setMouse({ x: e.clientX, y: e.clientY })
    window.addEventListener('mousemove', handler)
    return () => window.removeEventListener('mousemove', handler)
  }, [])

  return (
    <div ref={containerRef} className="relative min-h-screen overflow-hidden px-6 pb-16 pt-32 transition-all duration-500">
      <GridBackground mouseX={mouse.x} mouseY={mouse.y} />

      <div className="relative z-10 mx-auto max-w-7xl">
        <div className="grid items-center gap-12 lg:grid-cols-2 lg:gap-16">
          <div className="space-y-8">
            <h1 className="font-display text-4xl font-extrabold leading-[1.1] tracking-tight text-text-primary md:text-5xl lg:text-6xl">
              PREDICT THE NEXT CANDLE WITH ABSOLUTE VISION.
            </h1>
            <p className="max-w-lg text-lg leading-relaxed text-text-secondary">
              A dual-engine machine learning terminal built to chart trends, calculate sentiment, and protect equity.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link
                to="/signup"
                className="inline-flex animate-soft-pulse items-center rounded-full bg-accent px-8 py-4 text-sm font-semibold text-white transition-all duration-300 hover:brightness-110 active:scale-95"
              >
                Get Started Free
              </Link>
              <Link
                to="/login"
                className="inline-flex items-center rounded-full border border-pill bg-bento/50 px-8 py-4 text-sm font-semibold text-text-primary transition-all hover:bg-bento active:scale-95"
              >
                Log In
              </Link>
            </div>
          </div>

          <HeroChartTeaser />
        </div>

        <section className="mt-24">
          <h2 className="mb-8 font-display text-2xl font-bold text-text-primary">Core Capabilities</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {bentoFeatures.map(({ title, description, icon }) => {
              const Icon = iconMap[icon]
              return (
                <div
                  key={title}
                  className="card-surface group rounded-3xl p-6 light-shadow transition-all duration-300 hover:-translate-y-1"
                >
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/10">
                    <Icon className="h-6 w-6 text-accent" strokeWidth={1.5} />
                  </div>
                  <h3 className="mb-2 font-display text-lg font-bold text-text-primary">{title}</h3>
                  <p className="text-sm leading-relaxed text-text-secondary">{description}</p>
                </div>
              )
            })}
          </div>
        </section>
      </div>
    </div>
  )
}
