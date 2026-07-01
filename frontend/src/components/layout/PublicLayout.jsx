import { Outlet } from 'react-router-dom'
import PublicHeader from './PublicHeader'

export default function PublicLayout() {
  return (
    <div className="min-h-screen bg-canvas transition-all duration-500 ease-in-out">
      <PublicHeader />
      <Outlet />
    </div>
  )
}
