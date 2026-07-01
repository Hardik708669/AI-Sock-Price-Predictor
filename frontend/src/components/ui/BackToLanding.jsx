import { Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'

export default function BackToLanding() {
  return (
    <Link
      to="/"
      className="mb-6 inline-flex items-center gap-2 rounded-full bg-pill/40 px-4 py-2 text-sm font-medium text-text-secondary transition-all hover:bg-pill/60 hover:text-text-primary active:scale-95"
    >
      <ArrowLeft className="h-4 w-4" strokeWidth={1.5} />
      Back to Home
    </Link>
  )
}
