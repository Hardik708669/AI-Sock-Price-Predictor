const seededRandom = (seed) => {
  let s = seed
  return () => {
    s = (s * 16807) % 2147483647
    return (s - 1) / 2147483646
  }
}

export function generateStockHistory(days = 100, basePrice = 180, seed = 42) {
  const rand = seededRandom(seed)
  const data = []
  let price = basePrice
  const start = new Date()
  start.setDate(start.getDate() - days)

  for (let i = 0; i < days; i++) {
    const date = new Date(start)
    date.setDate(start.getDate() + i)
    const change = (rand() - 0.48) * 4.5
    price = Math.max(price + change, basePrice * 0.6)
    const confidence = 0.72 + rand() * 0.25
    data.push({
      date: date.toISOString().split('T')[0],
      price: +price.toFixed(2),
      confidence: +confidence.toFixed(2),
      type: 'history',
    })
  }
  return data
}

export function generatePrediction(lastPrice, days = 7, seed = 99) {
  const rand = seededRandom(seed)
  const data = []
  let price = lastPrice
  const start = new Date()

  for (let i = 1; i <= days; i++) {
    const date = new Date(start)
    date.setDate(start.getDate() + i)
    price += (rand() - 0.42) * 3.2
    data.push({
      date: date.toISOString().split('T')[0],
      price: +price.toFixed(2),
      confidence: +(0.68 + rand() * 0.22).toFixed(2),
      type: 'forecast',
    })
  }
  return data
}

export const watchlist = [
  { ticker: 'AAPL', name: 'Apple Inc.', signal: 'BUY', change: +2.34, price: 198.45, seed: 101 },
  { ticker: 'NVDA', name: 'NVIDIA Corp.', signal: 'BUY', change: +4.12, price: 892.3, seed: 202 },
  { ticker: 'TSLA', name: 'Tesla Inc.', signal: 'HOLD/SELL', change: -1.87, price: 248.67, seed: 303 },
  { ticker: 'BTC', name: 'Bitcoin', signal: 'BUY', change: +3.56, price: 67420.0, seed: 404 },
]

export const tickerMeta = {
  AAPL: { basePrice: 198, seed: 101, sector: 'Technology' },
  NVDA: { basePrice: 892, seed: 202, sector: 'Technology' },
  TSLA: { basePrice: 248, seed: 303, sector: 'Automotive' },
  BTC: { basePrice: 67420, seed: 404, sector: 'Crypto' },
}

export const bentoFeatures = [
  {
    title: 'Ensemble Hybrid Brain',
    description:
      'Dual-engine XGBoost + LSTM fusion model cross-validates every signal against 12 macro features before surfacing a trade recommendation.',
    icon: 'Brain',
  },
  {
    title: '7-Day Roadmap',
    description:
      'Projected price corridors with confidence bands visualize the next week of movement — not just a single target price.',
    icon: 'Route',
  },
  {
    title: 'Sentiment Radar',
    description:
      'Real-time NLP scoring of 200+ financial news sources feeds directly into the prediction weight matrix.',
    icon: 'Radar',
  },
  {
    title: 'Risk Guardrails',
    description:
      'Portfolio VaR, Sharpe ratio monitoring, and correlation heatmaps keep your equity protected at all times.',
    icon: 'Shield',
  },
]

export const newsFeed = [
  { headline: 'Fed signals potential rate cut in Q3, markets rally on dovish tone', impact: +0.75, source: 'Reuters' },
  { headline: 'NVIDIA unveils next-gen AI chip architecture, analysts raise targets', impact: +0.92, source: 'Bloomberg' },
  { headline: 'Tech sector faces antitrust scrutiny amid mega-cap consolidation', impact: -0.40, source: 'WSJ' },
  { headline: 'Oil prices surge 3% on Middle East supply concerns', impact: +0.55, source: 'CNBC' },
  { headline: 'Consumer confidence index drops below expectations for second month', impact: -0.62, source: 'FT' },
  { headline: 'Apple reports record services revenue, beats EPS estimates', impact: +0.88, source: 'MarketWatch' },
  { headline: 'Crypto ETF inflows hit all-time high as institutional adoption accelerates', impact: +0.71, source: 'CoinDesk' },
  { headline: 'Manufacturing PMI contracts for third consecutive quarter', impact: -0.48, source: 'Investing.com' },
  { headline: 'Tesla delivery numbers exceed Wall Street consensus by 8%', impact: +0.65, source: 'Barron\'s' },
  { headline: 'Global bond yields spike on inflation data surprise', impact: -0.55, source: 'Yahoo Finance' },
]

export const copilotMessages = [
  {
    role: 'assistant',
    content:
      'Good morning. I\'ve analyzed overnight market activity across your watchlist. Here\'s what stands out:',
    bullets: [
      { type: 'opportunity', text: 'NVDA showing bullish MACD crossover with RSI at 58 — momentum building without overbought conditions.' },
      { type: 'caution', text: 'TSLA sentiment score dropped -0.40 after regulatory headlines. Consider tightening stop-loss to $242.' },
      { type: 'opportunity', text: 'AAPL breaking above 50-day MA with volume 1.4× average — institutional accumulation pattern detected.' },
    ],
  },
  {
    role: 'user',
    content: 'What\'s the macro outlook for the next 48 hours?',
  },
  {
    role: 'assistant',
    content: 'Based on current data feeds and sentiment aggregation:',
    bullets: [
      { type: 'opportunity', text: 'Market mood index at 72/100 (Optimistic) — risk-on sentiment favors growth equities.' },
      { type: 'caution', text: 'VIX at 14.2 but rising — monitor for volatility spike around Thursday CPI release.' },
      { type: 'opportunity', text: 'Sector rotation signal: Technology → Energy transition probability at 34%.' },
    ],
  },
]

export const backtestAssets = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'BTC', 'SPY']

export const portfolioAllocation = [
  { name: 'Technology', value: 42, color: '#FF6B00' },
  { name: 'Energy', value: 18, color: '#00E676' },
  { name: 'Retail', value: 15, color: '#448AFF' },
  { name: 'Crypto', value: 12, color: '#E040FB' },
  { name: 'Healthcare', value: 8, color: '#FF1744' },
  { name: 'Cash', value: 5, color: '#8C9099' },
]

export const correlationMatrix = {
  assets: ['AAPL', 'NVDA', 'TSLA', 'BTC', 'XLE'],
  values: [
    [1.0, 0.72, 0.58, 0.31, 0.12],
    [0.72, 1.0, 0.65, 0.45, 0.08],
    [0.58, 0.65, 1.0, 0.52, 0.15],
    [0.31, 0.45, 0.52, 1.0, 0.22],
    [0.12, 0.08, 0.15, 0.22, 1.0],
  ],
}

export const volumeBars = Array.from({ length: 20 }, (_, i) => ({
  day: `D${i + 1}`,
  volume: 40 + Math.sin(i * 0.5) * 25 + (i % 3) * 8,
}))

export const macdData = Array.from({ length: 30 }, (_, i) => ({
  day: i,
  macd: Math.sin(i * 0.3) * 2.5,
  signal: Math.sin(i * 0.3 - 0.5) * 2,
  histogram: Math.sin(i * 0.3) * 0.8,
}))

export function getSentimentScore() {
  const total = newsFeed.reduce((sum, n) => sum + n.impact, 0)
  return Math.min(Math.max(50 + total * 8, 15), 95)
}

export function generateBacktestTimeline(points = 52) {
  const rand = seededRandom(777)
  const data = []
  let equity = 10000
  for (let i = 0; i < points; i++) {
    equity *= 1 + (rand() - 0.46) * 0.04
    data.push({ week: `W${i + 1}`, equity: +equity.toFixed(2), benchmark: +(10000 * (1 + i * 0.003)).toFixed(2) })
  }
  return data
}
