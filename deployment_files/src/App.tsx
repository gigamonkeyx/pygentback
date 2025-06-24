import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppLayout } from '@/components/layout/AppLayout';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { LoginPage } from '@/pages/LoginPage';
import { ReasoningPage } from '@/pages/ReasoningPage';
import { EvolutionPage } from '@/pages/EvolutionPage';
import { SearchPage } from '@/pages/SearchPage';
import { MonitoringPage } from '@/pages/MonitoringPage';
import { MCPMarketplacePage } from '@/pages/MCPMarketplacePage';
import { OllamaPage } from '@/pages/OllamaPage';
import { SettingsPage } from '@/pages/SettingsPage';
import ResearchAnalysisPage from '@/pages/ResearchAnalysisPage';
import { useAuth } from '@/stores/appStore';
import { ViewType } from '@/types';
import './globals.css';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
});

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuth();
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
};

const ViewRouter: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/chat" replace />} />
      <Route path="/chat" element={<ChatInterface />} />
      <Route path="/reasoning" element={<ReasoningPage />} />
      <Route path="/evolution" element={<EvolutionPage />} />
      <Route path="/search" element={<SearchPage />} />
      <Route path="/research-analysis" element={<ResearchAnalysisPage />} />
      <Route path="/monitoring" element={<MonitoringPage />} />
      <Route path="/mcp-marketplace" element={<MCPMarketplacePage />} />
      <Route path="/ollama" element={<OllamaPage />} />
      <Route path="/settings" element={<SettingsPage />} />
    </Routes>
  );
};

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-background font-sans antialiased">
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route
              path="/*"
              element={
                <ProtectedRoute>
                  <AppLayout>
                    <ViewRouter />
                  </AppLayout>
                </ProtectedRoute>
              }
            />
          </Routes>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
