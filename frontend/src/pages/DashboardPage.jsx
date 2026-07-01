import { useState } from 'react'
import { Activity, BarChart3, Gauge, Loader2, Sparkles, TrendingUp } from 'lucide-react'
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import StockChart from '../components/charts/StockChart'
import { useApp } from '../context/AppContext'
import { macdData, tickerMeta, volumeBars, watchlist } from '../data/mockData'

function RSIGauge({ value = 58 }) {
  const rotation = -90 + (value / 100) * 180
  const zone = value > 70 ? 'Overbought' : value < 30 ? 'Oversold' : 'Neutral'
  const zoneColor = value > 70 ? 'text-bearish' : value < 30 ? 'text-bullish' : 'text-accent'

  return (
    <div className="flex flex-col items-center">
      <div className="relative h-24 w-44 overflow-hidden">
        <div className="absolute inset-x-0 bottom-0 h-24 w-44 rounded-t-full border-[6px] border-pill border-b-0" />
        <div
          className="absolute bottom-0 left-1/2 h-20 w-1 origin-bottom rounded-full bg-accent transition-transform duration-700"
          style={{ transform: `translateX(-50%) rotate(${rotation}deg)` }}
        />
        <div className="absolute bottom-0 left-1/2 h-3 w-3 -translate-x-1/2 rounded-full bg-accent" />
      </div>
      <p className="mt-2 font-display text-2xl font-bold text-text-primary">{value}</p>
      <p className={`text-xs font-medium ${zoneColor}`}>{zone}</p>
    </div>
  )
}

export default function DashboardPage() {
  const {
    selectedTicker,
    setSelectedTicker,
    predictionVisible,
    setPredictionVisible,
    calibrationState,
    setCalibrationState,
  } = useApp()
  const [predicting, setPredicting] = useState(false)
  const meta = tickerMeta[selectedTicker]

  const handlePredict = () => {
    setPredicting(true)
    setTimeout(() => {
      setPredictionVisible(true)
      setPredicting(false)
    }, 1200)
  }

  const handleCalibrate = () => {
    setCalibrationState('running')
    setTimeout(() => setCalibrationState('optimized'), 2500)
  }

  return (
    <div className="transition-all duration-500">
      <div className="mx-auto grid max-w-[1600px] gap-6 lg:grid-cols-12">
        {/* Center stage */}
        <div className="space-y-6 lg:col-span-9">
          <div className="card-surface rounded-3xl p-6 light-shadow">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
              <div>
                <h1 className="font-display text-2xl font-bold text-text-primary">{selectedTicker}</h1>
                <p className="text-sm text-text-secondary">{meta?.sector} · 100-Day History</p>
              </div>
              <button
                onClick={handlePredict}
                disabled={predicting}
                className="flex items-center gap-2 rounded-full bg-accent px-6 py-3 text-sm font-semibold text-white transition-all hover:brightness-110 active:scale-95 disabled:opacity-70"
              >
                {predicting ? (
                  <Loader2 className="h-4 w-4 animate-spin" strokeWidth={1.5} />
                ) : (
                  <Sparkles className="h-4 w-4" strokeWidth={1.5} />
                )}
                {predicting ? 'Computing...' : 'Predict'}
              </button>
            </div>
            <div className="h-[420px]">
              <StockChart ticker={selectedTicker} showPrediction={predictionVisible} />
            </div>
            {predictionVisible && (
              <p className="mt-3 text-center text-xs text-text-secondary">
                Dotted line = 7-day AI forecast · Hover points for confidence scores
              </p>
            )}
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            {/* Auto-ML Reactor */}
            <div className="card-surface rounded-3xl p-6 light-shadow">
              <div className="mb-4 flex items-center gap-2">
                <Activity className="h-5 w-5 text-accent" strokeWidth={1.5} />
                <h3 className="font-display font-bold text-text-primary">Auto-ML Parameter Tuning</h3>
              </div>
              <p className="mb-4 text-sm text-text-secondary">
                Hyperparameter search optimizing learning rate, estimators, and max depth via Optuna framework.
              </p>
              <div className="mb-4 flex gap-4 text-xs text-text-secondary">
                <span>LR: 0.08</span>
                <span>Estimators: 142</span>
                <span>Depth: 6</span>
              </div>
              <button
                onClick={handleCalibrate}
                disabled={calibrationState === 'running'}
                className="flex w-full items-center justify-center gap-2 rounded-full bg-pill px-4 py-3 text-sm font-medium text-text-primary transition-all hover:bg-accent/10 active:scale-95 disabled:opacity-60"
              >
                {calibrationState === 'running' ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin-slow text-accent" strokeWidth={1.5} />
                    Calibrating Framework...
                  </>
                ) : (
                  'Calibrate Framework'
                )}
              </button>
              {calibrationState === 'optimized' && (
                <span className="mt-3 inline-flex rounded-full bg-bullish/10 px-3 py-1 text-xs font-semibold text-bullish">
                  ✓ Optimized State
                </span>
              )}
            </div>

            {/* Clue Indicator Deck */}
            <div className="card-surface rounded-3xl p-6 light-shadow">
              <div className="mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-accent" strokeWidth={1.5} />
                <h3 className="font-display font-bold text-text-primary">Clue Indicator Deck</h3>
              </div>

              <p className="mb-2 text-xs font-medium uppercase tracking-wider text-text-secondary">Volume</p>
              <ResponsiveContainer width="100%" height={80}>
                <BarChart data={volumeBars}>
                  <XAxis dataKey="day" hide />
                  <YAxis hide />
                  <Bar dataKey="volume" radius={[4, 4, 0, 0]}>
                    {volumeBars.map((_, i) => (
                      <Cell key={i} fill={i % 2 === 0 ? '#DE5D00' : '#B8BBC2'} fillOpacity={0.7} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <p className="mb-2 flex items-center gap-1 text-xs font-medium uppercase tracking-wider text-text-secondary">
                    <Gauge className="h-3 w-3" strokeWidth={1.5} /> RSI
                  </p>
                  <RSIGauge value={58} />
                </div>
                <div>
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-text-secondary">MACD</p>
                  <ResponsiveContainer width="100%" height={100}>
                    <BarChart data={macdData.slice(-12)}>
                      <Tooltip
                        content={({ active, payload }) =>
                          active && payload?.[0] ? (
                            <div className="rounded-lg bg-bento px-2 py-1 text-xs light-shadow">
                              {payload[0].value?.toFixed(2)}
                            </div>
                          ) : null
                        }
                      />
                      <Bar dataKey="histogram">
                        {macdData.slice(-12).map((entry, i) => (
                          <Cell key={i} fill={entry.histogram >= 0 ? '#2E7D32' : '#C62828'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="mt-1 text-xs text-bullish">Bullish crossover detected</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right watchlist rail */}
        <aside className="lg:col-span-3">
          <div className="card-surface sticky top-24 rounded-3xl p-6 light-shadow">
            <div className="mb-4 flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-accent" strokeWidth={1.5} />
              <h3 className="font-display font-bold text-text-primary">Watchlist Signals</h3>
            </div>
            <div className="space-y-3">
              {watchlist.map((item) => (
                <button
                  key={item.ticker}
                  onClick={() => {
                    setSelectedTicker(item.ticker)
                    setPredictionVisible(false)
                  }}
                  className={`w-full rounded-2xl p-4 text-left transition-all active:scale-95 ${
                    selectedTicker === item.ticker
                      ? 'bg-accent/10 ring-1 ring-accent/30'
                      : 'bg-pill/30 hover:bg-pill/50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-text-primary">{item.ticker}</p>
                      <p className="text-xs text-text-secondary">{item.name}</p>
                    </div>
                    <p className={`text-sm font-medium ${item.change >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                      {item.change >= 0 ? '+' : ''}
                      {item.change}%
                    </p>
                  </div>
                  <div className="mt-2 flex items-center justify-between">
                    <span className="text-sm text-text-secondary">${item.price.toLocaleString()}</span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-bold tracking-wider ${
                        item.signal === 'BUY'
                          ? 'bg-bullish/10 text-bullish shadow-[0_0_12px_rgba(46,125,50,0.3)]'
                          : 'bg-bearish/10 text-bearish'
                      }`}
                    >
                      [ {item.signal === 'BUY' ? 'BUY SIGNAL' : 'HOLD/SELL'} ]
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
