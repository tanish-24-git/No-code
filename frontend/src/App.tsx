import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { FlaskConical, Folder, Sparkles, LayoutGrid, Plus } from 'lucide-react';
import Playground from './pages/Playground';
import Compare from './pages/Compare';
import Projects from './pages/Projects';
import AIChat from './pages/AIChat';
import CreateJob from './pages/CreateJob';
import './index.css';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-background">
          {/* Navigation */}
          <nav className="glass border-b border-border sticky top-0 z-40 backdrop-blur-xl">
            <div className="max-w-7xl mx-auto px-6 py-4">
              <div className="flex items-center justify-between">
                <Link to="/" className="flex items-center gap-3 group">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center glow-primary">
                    <FlaskConical className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h1 className="text-xl font-bold gradient-text">
                      LLM Fine-Tuning Platform
                    </h1>
                    <p className="text-xs text-muted-foreground">Industrial-grade AI training</p>
                  </div>
                </Link>

                <div className="flex items-center gap-2">
                  <Link to="/projects" className="flex items-center gap-2 px-4 py-2 rounded-xl hover:bg-secondary transition-all text-sm font-medium">
                    <Folder className="w-4 h-4" />
                    Projects
                  </Link>
                  <Link to="/create-job" className="flex items-center gap-2 px-4 py-2 rounded-xl bg-primary text-primary-foreground hover:opacity-90 transition-all text-sm font-medium glow-primary">
                    <Plus className="w-4 h-4" />
                    New Job
                  </Link>
                  <Link to="/" className="flex items-center gap-2 px-4 py-2 rounded-xl hover:bg-secondary transition-all text-sm font-medium">
                    <LayoutGrid className="w-4 h-4" />
                    Playground
                  </Link>
                  <Link to="/ai-chat" className="flex items-center gap-2 px-4 py-2 rounded-xl hover:bg-secondary transition-all text-sm font-medium">
                    <Sparkles className="w-4 h-4" />
                    AI Assistant
                  </Link>
                </div>
              </div>
            </div>
          </nav>

          {/* Routes */}
          <Routes>
            <Route path="/" element={<Playground />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/projects" element={<Projects />} />
            <Route path="/create-job" element={<CreateJob />} />
            <Route path="/ai-chat" element={<AIChat />} />
          </Routes>

          <Toaster
            position="bottom-right"
            toastOptions={{
              className: 'glass border border-border',
              style: {
                background: 'hsl(var(--card))',
                color: 'hsl(var(--foreground))',
              },
            }}
          />
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
