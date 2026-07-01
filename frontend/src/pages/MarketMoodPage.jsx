import { useState } from 'react'
import { AlertTriangle, ArrowUpRight, Bot, Send } from 'lucide-react'
import { copilotMessages, getSentimentScore, newsFeed } from '../data/mockData'

function SentimentSpeedometer({ score }) {
  const rotation = -90 + (score / 100) * 180
  const label = score < 35 ? 'Fear' : score > 65 ? 'Highly Optimistic' : 'Neutral'
  const labelColor = score < 35 ? 'text-bearish' : score > 65 ? 'text-bullish' : 'text-accent'

  return (
    <div className="card-surface rounded-3xl p-8 light-shadow">
      <h2 className="mb-6 font-display text-xl font-bold text-text-primary">Market Sentiment Speedometer</h2>
      <div className="relative mx-auto flex h-48 w-full max-w-xs flex-col items-center justify-end">
        <svg viewBox="0 0 200 110" className="w-full">
          <defs>
            <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#C62828" />
              <stop offset="50%" stopColor="#DE5D00" />
              <stop offset="100%" stopColor="#2E7D32" />
            </linearGradient>
          </defs>
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="url(#gaugeGrad)"
            strokeWidth="12"
            strokeLinecap="round"
          />
          <line
            x1="100"
            y1="100"
            x2={100 + 70 * Math.cos((rotation * Math.PI) / 180)}
            y2={100 + 70 * Math.sin((rotation * Math.PI) / 180)}
            stroke="var(--color-text-primary)"
            strokeWidth="3"
            strokeLinecap="round"
            className="transition-all duration-700"
          />
          <circle cx="100" cy="100" r="6" fill="var(--color-accent)" />
        </svg>
        <p className="font-display text-4xl font-bold text-text-primary">{Math.round(score)}</p>
        <p className={`mt-1 text-sm font-semibold ${labelColor}`}>{label}</p>
      </div>
    </div>
  )
}

function NewsStream() {
  return (
    <div className="card-surface mt-6 rounded-3xl p-6 light-shadow">
      <h3 className="mb-4 font-display text-lg font-bold text-text-primary">Live News Feed</h3>
      <div className="max-h-[420px] space-y-3 overflow-y-auto pr-2">
        {newsFeed.map((item, i) => (
          <div
            key={i}
            className="rounded-2xl bg-pill/30 p-4 transition-all hover:bg-pill/50"
          >
            <div className="mb-2 flex items-start justify-between gap-3">
              <p className="text-sm leading-snug text-text-primary">{item.headline}</p>
              <span
                className={`shrink-0 rounded-full px-2 py-0.5 font-mono text-xs font-bold ${
                  item.impact >= 0 ? 'bg-bullish/10 text-bullish' : 'bg-bearish/10 text-bearish'
                }`}
              >
                {item.impact >= 0 ? '+' : ''}
                {item.impact.toFixed(2)}
              </span>
            </div>
            <p className="text-xs text-text-secondary">{item.source}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

function CopilotPanel() {
  const [messages, setMessages] = useState(copilotMessages)
  const [input, setInput] = useState('')

  const handleSend = (e) => {
    e.preventDefault()
    if (!input.trim()) return
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: input },
      {
        role: 'assistant',
        content: 'Analyzing your query against current market data...',
        bullets: [
          { type: 'opportunity', text: 'Cross-referencing 847 data points from your watchlist and macro indicators.' },
          { type: 'caution', text: 'Elevated correlation between NVDA and BTC suggests reduced diversification benefit.' },
        ],
      },
    ])
    setInput('')
  }

  return (
    <div className="card-surface flex h-full min-h-[640px] flex-col rounded-3xl light-shadow">
      <div className="flex items-center gap-3 border-b border-pill/30 px-6 py-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-accent/10">
          <Bot className="h-5 w-5 text-accent" strokeWidth={1.5} />
        </div>
        <div>
          <h2 className="font-display text-lg font-bold text-text-primary">Trade Vision Copilot</h2>
          <p className="text-xs text-text-secondary">AI-powered market intelligence</p>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-6 py-6">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-accent text-white'
                  : 'bg-pill/40 text-text-primary'
              }`}
            >
              <p className="text-sm leading-relaxed">{msg.content}</p>
              {msg.bullets && (
                <ul className="mt-3 space-y-2">
                  {msg.bullets.map((b, j) => (
                    <li key={j} className="flex items-start gap-2 text-sm">
                      {b.type === 'caution' ? (
                        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-bearish" strokeWidth={1.5} />
                      ) : (
                        <ArrowUpRight className="mt-0.5 h-4 w-4 shrink-0 text-bullish" strokeWidth={1.5} />
                      )}
                      <span>{b.text}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSend} className="border-t border-pill/30 p-4">
        <div className="flex items-center gap-3 rounded-full bg-pill/40 px-4 py-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask Trade Vision Copilot about current market anomalies..."
            className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-secondary"
          />
          <button
            type="submit"
            className="flex h-9 w-9 items-center justify-center rounded-full bg-accent text-white transition-all active:scale-95"
          >
            <Send className="h-4 w-4" strokeWidth={1.5} />
          </button>
        </div>
      </form>
    </div>
  )
}

export default function MarketMoodPage() {
  const sentimentScore = getSentimentScore()

  return (
    <div className="mx-auto max-w-7xl transition-all duration-500">
      <div className="grid gap-6 lg:grid-cols-2">
          <div>
            <SentimentSpeedometer score={sentimentScore} />
            <NewsStream />
          </div>
        <CopilotPanel />
      </div>
    </div>
  )
}
