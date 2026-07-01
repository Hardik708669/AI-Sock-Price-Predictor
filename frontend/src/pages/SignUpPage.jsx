import { useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { Eye, EyeOff, Lock, Mail, User } from 'lucide-react'
import BackToLanding from '../components/ui/BackToLanding'
import { useAuth } from '../context/AuthContext'

export default function SignUpPage() {
  const { signup, isAuthenticated } = useAuth()
  const navigate = useNavigate()

  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  if (isAuthenticated) {
    return <Navigate to="/app/dashboard" replace />
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    setError('')

    if (!name || !email || !password) {
      setError('Please fill in all fields.')
      return
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.')
      return
    }

    setLoading(true)
    setTimeout(() => {
      signup(name, email, password)
      setLoading(false)
      navigate('/app/dashboard', { replace: true })
    }, 800)
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-6 pt-20 pb-12">
      <div className="w-full max-w-md">
        <BackToLanding />
        <div className="mb-8 text-center">
          <h1 className="font-display text-3xl font-bold text-text-primary">Create your account</h1>
          <p className="mt-2 text-text-secondary">Start predicting markets with AI-powered vision</p>
        </div>

        <form onSubmit={handleSubmit} className="card-surface space-y-5 rounded-3xl p-8 light-shadow">
          {error && (
            <div className="rounded-2xl bg-bearish/10 px-4 py-3 text-sm text-bearish">{error}</div>
          )}

          <div>
            <label className="mb-2 block text-sm font-medium text-text-secondary">Full Name</label>
            <div className="flex items-center gap-3 rounded-2xl bg-pill/30 px-4 py-3">
              <User className="h-4 w-4 text-text-secondary" strokeWidth={1.5} />
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Alex Morgan"
                className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-secondary/60"
              />
            </div>
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-text-secondary">Email</label>
            <div className="flex items-center gap-3 rounded-2xl bg-pill/30 px-4 py-3">
              <Mail className="h-4 w-4 text-text-secondary" strokeWidth={1.5} />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@company.com"
                className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-secondary/60"
              />
            </div>
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-text-secondary">Password</label>
            <div className="flex items-center gap-3 rounded-2xl bg-pill/30 px-4 py-3">
              <Lock className="h-4 w-4 text-text-secondary" strokeWidth={1.5} />
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Min. 6 characters"
                className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-secondary/60"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="text-text-secondary active:scale-95"
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" strokeWidth={1.5} />
                ) : (
                  <Eye className="h-4 w-4" strokeWidth={1.5} />
                )}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-full bg-accent py-3.5 text-sm font-semibold text-white transition-all hover:brightness-110 active:scale-95 disabled:opacity-70"
          >
            {loading ? 'Creating account...' : 'Create Account'}
          </button>
        </form>

        <p className="mt-6 text-center text-sm text-text-secondary">
          Already have an account?{' '}
          <Link to="/login" className="font-semibold text-accent hover:underline">
            Log in
          </Link>
        </p>
      </div>
    </div>
  )
}
