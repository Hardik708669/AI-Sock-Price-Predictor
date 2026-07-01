import { useEffect, useState } from 'react'
import { Calendar, DollarSign, LineChart, Play, TrendingUp } from 'lucide-react'
import {
  Area,
  AreaChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { backtestAssets, generateBacktestTimeline } from '../data/mockData'
import { useTheme } from '../context/ThemeContext'

export default function BacktesterPage() {
  const { isDark } = useTheme()
  const [asset, setAsset] = useState('NVDA')
  const [startDate, setStartDate] = useState('2024-01-01')
  const [endDate, setEndDate] = useState('2025-01-01')
  const [capital, setCapital] = useState(10000)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [timelineData, setTimelineData] = useState([])
  const [complete, setComplete] = useState(false)

  const fullTimeline = generateBacktestTimeline(52)

  useEffect(() => {
    if (!running) return
    setTimelineData([])
    setProgress(0)
    setComplete(false)

    let step = 0
    const interval = setInterval(() => {
      step++
      setProgress((step / 52) * 100)
      setTimelineData(fullTimeline.slice(0, step))
      if (step >= 52) {
        clearInterval(interval)
        setRunning(false)
        setComplete(true)
      }
    }, 60)

    return () => clearInterval(interval)
  }, [running])

  const winLossData = [
    { name: 'Wins', value: 68, color: isDark ? '#00E676' : '#2E7D32' },
    { name: 'Losses', value: 32, color: isDark ? '#FF1744' : '#C62828' },
  ]

  const handleRun = () => {
    setRunning(true)
  }

  const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)'
  const accentColor = isDark ? '#FF6B00' : '#DE5D00'

  return (
    <div className="mx-auto max-w-7xl space-y-8 transition-all duration-500">
        {/* Config panel */}
        <div className="card-surface rounded-3xl p-8 light-shadow">
          <h2 className="mb-6 font-display text-lg font-bold text-text-primary">Historical Configurator</h2>
          <div className="flex flex-wrap items-end gap-4">
            <div className="rounded-full bg-pill/40 px-4 py-2">
              <label className="mb-1 block text-xs text-text-secondary">Asset</label>
              <select
                value={asset}
                onChange={(e) => setAsset(e.target.value)}
                className="bg-transparent text-sm font-medium text-text-primary outline-none"
              >
                {backtestAssets.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2 rounded-full bg-pill/40 px-4 py-2">
              <Calendar className="h-4 w-4 text-text-secondary" strokeWidth={1.5} />
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="bg-transparent text-sm text-text-primary outline-none"
              />
              <span className="text-text-secondary">→</span>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="bg-transparent text-sm text-text-primary outline-none"
              />
            </div>

            <div className="flex items-center gap-2 rounded-full bg-pill/40 px-4 py-2">
              <DollarSign className="h-4 w-4 text-text-secondary" strokeWidth={1.5} />
              <input
                type="number"
                value={capital}
                onChange={(e) => setCapital(Number(e.target.value))}
                className="w-28 bg-transparent text-sm font-medium text-text-primary outline-none"
              />
            </div>

            <button
              onClick={handleRun}
              disabled={running}
              className="flex items-center gap-2 rounded-full bg-accent px-8 py-3 text-sm font-semibold text-white transition-all hover:brightness-110 active:scale-95 disabled:opacity-60"
            >
              <Play className="h-4 w-4" strokeWidth={1.5} />
              Initialize Backtest Simulation
            </button>
          </div>
        </div>

        {/* Progress bar */}
        {(running || complete) && (
          <div className="space-y-2">
            <div className="h-2 overflow-hidden rounded-full bg-pill/40">
              <div
                className="h-full rounded-full bg-accent transition-all duration-100"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-center text-xs text-text-secondary">
              {running ? `Simulating ${asset} · ${Math.round(progress)}% complete` : 'Simulation complete'}
            </p>
          </div>
        )}

        {/* Runway chart */}
        <div className="card-surface rounded-3xl p-6 light-shadow">
          <h3 className="mb-4 font-display font-bold text-text-primary">Simulated Runway</h3>
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={timelineData.length ? timelineData : [{ week: 'W0', equity: capital, benchmark: capital }]}>
              <defs>
                <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={accentColor} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={accentColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="week" tick={{ fill: isDark ? '#8C9099' : '#5A5D64', fontSize: 11 }} />
              <YAxis tick={{ fill: isDark ? '#8C9099' : '#5A5D64', fontSize: 11 }} />
              <Tooltip
                content={({ active, payload }) =>
                  active && payload?.[0] ? (
                    <div className="rounded-xl bg-bento px-3 py-2 text-xs light-shadow">
                      <p className="font-semibold">${payload[0].value?.toLocaleString()}</p>
                    </div>
                  ) : null
                }
              />
              <Area
                type="monotone"
                dataKey="equity"
                stroke={accentColor}
                strokeWidth={2}
                fill="url(#equityGrad)"
                dot={false}
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="benchmark"
                stroke={isDark ? '#8C9099' : '#5A5D64'}
                strokeWidth={1}
                strokeDasharray="4 4"
                fill="none"
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Flight report */}
        {complete && (
          <div className="grid gap-6 md:grid-cols-3">
            <div className="card-surface rounded-3xl p-8 text-center light-shadow">
              <TrendingUp className="mx-auto mb-4 h-8 w-8 text-bullish" strokeWidth={1.5} />
              <p className="text-sm text-text-secondary">Total ROI</p>
              <p className="font-display text-5xl font-bold text-bullish">+42.3%</p>
              <p className="mt-1 text-sm text-text-secondary">Net Yield</p>
            </div>

            <div className="card-surface rounded-3xl p-8 light-shadow">
              <p className="mb-4 text-center text-sm text-text-secondary">Win/Loss Ratio</p>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie
                    data={winLossData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={70}
                    dataKey="value"
                    stroke="none"
                  >
                    {winLossData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <p className="text-center text-sm text-text-primary">
                <span className="font-bold text-bullish">68%</span> wins ·{' '}
                <span className="font-bold text-bearish">32%</span> losses
              </p>
            </div>

            <div className="card-surface rounded-3xl p-8 light-shadow">
              <p className="mb-2 text-sm text-text-secondary">Maximum Drawdown Index</p>
              <p className="font-display text-3xl font-bold text-bearish">-12.8%</p>
              <div className="mt-4 h-3 overflow-hidden rounded-full bg-pill/40">
                <div className="h-full w-[28%] rounded-full bg-gradient-to-r from-bearish/60 to-bearish" />
              </div>
              <p className="mt-3 text-xs leading-relaxed text-text-secondary">
                Severe dip encountered at week 34. Risk guardrails would have triggered a 15% position reduction.
              </p>
            </div>
          </div>
        )}
    </div>
  )
}
