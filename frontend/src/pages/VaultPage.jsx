import { Fragment } from 'react'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { PieChart as PieIcon, Shield, AlertCircle } from 'lucide-react'
import { correlationMatrix, portfolioAllocation } from '../data/mockData'
import { useTheme } from '../context/ThemeContext'

function AllocationDonut() {
  const { isDark } = useTheme()

  return (
    <div className="card-surface rounded-3xl p-8 light-shadow">
      <div className="mb-6 flex items-center gap-2">
        <PieIcon className="h-5 w-5 text-accent" strokeWidth={1.5} />
        <h2 className="font-display text-xl font-bold text-text-primary">Asset Allocation</h2>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie
            data={portfolioAllocation}
            cx="50%"
            cy="50%"
            innerRadius={70}
            outerRadius={110}
            paddingAngle={3}
            dataKey="value"
            stroke="none"
          >
            {portfolioAllocation.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            content={({ active, payload }) =>
              active && payload?.[0] ? (
                <div className="rounded-xl bg-bento px-3 py-2 text-xs light-shadow">
                  <p className="font-semibold text-text-primary">{payload[0].name}</p>
                  <p className="text-text-secondary">{payload[0].value}%</p>
                </div>
              ) : null
            }
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="mt-4 flex flex-wrap justify-center gap-3">
        {portfolioAllocation.map((item) => (
          <div key={item.name} className="flex items-center gap-2 text-xs">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
            <span className="text-text-secondary">{item.name}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function CorrelationMatrix() {
  const { assets, values } = correlationMatrix

  const getColor = (val) => {
    if (val >= 0.7) return 'bg-amber-400/40 ring-1 ring-amber-400/60'
    if (val >= 0.4) return 'bg-accent/15'
    return 'bg-pill/30'
  }

  return (
    <div className="card-surface rounded-3xl p-8 light-shadow">
      <h2 className="mb-6 font-display text-xl font-bold text-text-primary">Risk Correlation Matrix</h2>
      <div className="overflow-x-auto">
        <div
          className="inline-grid gap-1"
          style={{ gridTemplateColumns: `80px repeat(${assets.length}, 1fr)` }}
        >
          <div />
          {assets.map((a) => (
            <div key={a} className="px-1 text-center text-xs font-medium text-text-secondary">
              {a}
            </div>
          ))}
          {assets.map((rowAsset, i) => (
            <Fragment key={rowAsset}>
              <div className="flex items-center text-xs font-medium text-text-secondary">{rowAsset}</div>
              {values[i].map((val, j) => (
                <div
                  key={`${rowAsset}-${assets[j]}`}
                  className={`flex h-12 items-center justify-center rounded-lg text-xs font-mono font-medium text-text-primary transition-colors ${getColor(val)}`}
                  title={`${rowAsset} × ${assets[j]}: ${val.toFixed(2)}`}
                >
                  {val.toFixed(2)}
                </div>
              ))}
            </Fragment>
          ))}
        </div>
      </div>
      <p className="mt-4 flex items-center gap-2 text-xs text-text-secondary">
        <span className="inline-block h-3 w-3 rounded bg-amber-400/40 ring-1 ring-amber-400/60" />
        Amber zones indicate overlapping volatility — limited diversification benefit
      </p>
    </div>
  )
}

function PortfolioHealthBar() {
  const sharpeRatio = 1.84
  const var95 = 4.2
  const vulnerability = 32

  return (
    <div className="card-surface rounded-3xl p-8 light-shadow">
      <div className="mb-6 flex items-center gap-2">
        <Shield className="h-5 w-5 text-accent" strokeWidth={1.5} />
        <h2 className="font-display text-xl font-bold text-text-primary">Sharpe Portfolio Health</h2>
      </div>

      <div className="mb-6 grid gap-6 sm:grid-cols-3">
        <div>
          <p className="text-xs uppercase tracking-wider text-text-secondary">Sharpe Ratio</p>
          <p className="font-display text-3xl font-bold text-bullish">{sharpeRatio}</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-text-secondary">VaR (95%)</p>
          <p className="font-display text-3xl font-bold text-text-primary">{var95}%</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wider text-text-secondary">Vulnerability</p>
          <p className="font-display text-3xl font-bold text-accent">{vulnerability}/100</p>
        </div>
      </div>

      <p className="mb-2 text-sm font-medium text-text-primary">Portfolio Vulnerability Bar</p>
      <div className="h-4 overflow-hidden rounded-full bg-pill/40">
        <div
          className="h-full rounded-full bg-gradient-to-r from-bullish via-accent to-bearish transition-all duration-700"
          style={{ width: `${vulnerability}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-xs text-text-secondary">
        <span>Protected</span>
        <span>At Risk</span>
      </div>

      <div className="mt-6 flex items-start gap-3 rounded-2xl bg-pill/30 p-4">
        <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-accent" strokeWidth={1.5} />
        <div>
          <p className="text-sm font-medium text-text-primary">Diversification Recommendation</p>
          <p className="mt-1 text-sm leading-relaxed text-text-secondary">
            Your Technology + Crypto overlap (0.45 correlation) concentrates risk. Consider reallocating 8% from
            NVDA into Energy (XLE) or Healthcare (XLV) to improve your Sharpe ratio and reduce VaR exposure.
          </p>
        </div>
      </div>
    </div>
  )
}

export default function VaultPage() {
  return (
    <div className="mx-auto max-w-7xl space-y-8 transition-all duration-500">
        <div className="grid gap-6 lg:grid-cols-2">
          <AllocationDonut />
          <CorrelationMatrix />
        </div>

      <PortfolioHealthBar />
    </div>
  )
}
