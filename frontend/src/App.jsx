import { BrowserRouter, Navigate, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import { AuthProvider } from './context/AuthContext'
import { AppProvider } from './context/AppContext'
import ProtectedRoute from './components/auth/ProtectedRoute'
import PublicLayout from './components/layout/PublicLayout'
import AppLayout from './components/layout/AppLayout'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import SignUpPage from './pages/SignUpPage'
import DashboardPage from './pages/DashboardPage'
import MarketMoodPage from './pages/MarketMoodPage'
import BacktesterPage from './pages/BacktesterPage'
import VaultPage from './pages/VaultPage'

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <AppProvider>
          <BrowserRouter>
            <Routes>
              {/* Public routes — landing, login, signup */}
              <Route element={<PublicLayout />}>
                <Route path="/" element={<LandingPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/signup" element={<SignUpPage />} />
              </Route>

              {/* Protected app routes — sidebar layout */}
              <Route
                path="/app"
                element={
                  <ProtectedRoute>
                    <AppLayout />
                  </ProtectedRoute>
                }
              >
                <Route index element={<Navigate to="dashboard" replace />} />
                <Route path="dashboard" element={<DashboardPage />} />
                <Route path="market-mood" element={<MarketMoodPage />} />
                <Route path="backtester" element={<BacktesterPage />} />
                <Route path="vault" element={<VaultPage />} />
              </Route>

              {/* Legacy redirects */}
              <Route path="/dashboard" element={<Navigate to="/app/dashboard" replace />} />
              <Route path="/market-mood" element={<Navigate to="/app/market-mood" replace />} />
              <Route path="/backtester" element={<Navigate to="/app/backtester" replace />} />
              <Route path="/vault" element={<Navigate to="/app/vault" replace />} />

              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </BrowserRouter>
        </AppProvider>
      </AuthProvider>
    </ThemeProvider>
  )
}
