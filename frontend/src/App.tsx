import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import { Zap, GitCompare } from 'lucide-react'
import Playground from './pages/Playground'
import Compare from './pages/Compare'
import './index.css'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-background">
          {/* Navigation */}
          <nav className="border-b border-border bg-card/50 backdrop-blur-md sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-8 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-6 h-6 text-primary" />
                  <h1 className="text-xl font-bold">LLM Platform</h1>
                </div>
                <div className="flex gap-4">
                  <Link
                    to="/"
                    className="px-4 py-2 rounded-md hover:bg-accent transition-colors"
                  >
                    Playground
                  </Link>
                  <Link
                    to="/compare"
                    className="px-4 py-2 rounded-md hover:bg-accent transition-colors flex items-center gap-2"
                  >
                    <GitCompare className="w-4 h-4" />
                    Compare
                  </Link>
                </div>
              </div>
            </div>
          </nav>

          {/* Routes */}
          <Routes>
            <Route path="/" element={<Playground />} />
            <Route path="/compare" element={<Compare />} />
          </Routes>

          {/* Toast Notifications */}
          <Toaster position="bottom-right" />
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
