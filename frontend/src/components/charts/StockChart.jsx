import { useMemo } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { generatePrediction, generateStockHistory, tickerMeta } from '../../data/mockData'
import { useTheme } from '../../context/ThemeContext'

export default function StockChart({ ticker, showPrediction }) {
  const { isDark } = useTheme()
  const meta = tickerMeta[ticker] || tickerMeta.AAPL

  const { historyData, forecastData } = useMemo(() => {
    const history = generateStockHistory(100, meta.basePrice, meta.seed)
    if (!showPrediction) return { historyData: history, forecastData: [] }

    const lastPoint = history[history.length - 1]
    const forecast = generatePrediction(lastPoint.price, 7, meta.seed + 50)
    const forecastWithBridge = [lastPoint, ...forecast]
    return { historyData: history, forecastData: forecastWithBridge }
  }, [ticker, showPrediction, meta])

  const lineColor = isDark ? '#F3F3F5' : '#1E2022'
  const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)'
  const accentColor = isDark ? '#FF6B00' : '#DE5D00'

  const mergedForAxis = showPrediction ? [...historyData, ...forecastData.slice(1)] : historyData

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={mergedForAxis} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fill: isDark ? '#8C9099' : '#5A5D64', fontSize: 11 }}
          tickFormatter={(v) => v.slice(5)}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={['auto', 'auto']}
          tick={{ fill: isDark ? '#8C9099' : '#5A5D64', fontSize: 11 }}
          tickFormatter={(v) => `$${v}`}
          width={70}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.[0]) return null
            const d = payload[0].payload
            return (
              <div className="rounded-xl bg-bento px-4 py-3 text-sm light-shadow">
                <p className="font-semibold text-text-primary">${d.price?.toFixed(2)}</p>
                <p className="text-text-secondary">{d.date}</p>
                {d.confidence && (
                  <p className="mt-1 text-xs text-accent">AI Confidence: {(d.confidence * 100).toFixed(0)}%</p>
                )}
              </div>
            )
          }}
        />
        <Line
          type="monotone"
          dataKey="price"
          data={historyData}
          stroke={lineColor}
          strokeWidth={2}
          dot={false}
          connectNulls
        />
        {showPrediction && forecastData.length > 0 && (
          <Line
            type="monotone"
            dataKey="price"
            data={forecastData}
            stroke={accentColor}
            strokeWidth={2.5}
            dot={false}
            strokeDasharray="8 5"
            connectNulls
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  )
}
