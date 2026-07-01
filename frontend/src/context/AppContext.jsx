import { createContext, useContext, useState } from 'react'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [selectedTicker, setSelectedTicker] = useState('AAPL')
  const [predictionVisible, setPredictionVisible] = useState(false)
  const [calibrationState, setCalibrationState] = useState('idle')

  return (
    <AppContext.Provider
      value={{
        selectedTicker,
        setSelectedTicker,
        predictionVisible,
        setPredictionVisible,
        calibrationState,
        setCalibrationState,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
