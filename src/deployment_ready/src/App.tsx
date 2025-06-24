import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppLayout } from '@/components/layout/AppLayout';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { ReasoningPage } from '@/pages/ReasoningPage';
import { MonitoringPage } from '@/pages/MonitoringPage';
import { MCPMarketplacePage } from '@/pages/MCPMarketplacePage';
import { SettingsPage } from '@/pages/SettingsPage';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { websocketService } from '@/services/websocket';
import { useAppStore } from '@/stores/appStore';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      cacheTime: 1000 * 60 * 10, // 10 minutes
    },
  },
});

const ViewRouter: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/chat" replace />} />
      <Route path="/chat" element={<ChatInterface />} />
      <Route path="/reasoning" element={<ReasoningPage />} />
      <Route path="/monitoring" element={<MonitoringPage />} />
      <Route path="/mcp-marketplace" element={<MCPMarketplacePage />} />
      <Route path="/settings" element={<SettingsPage />} />
    </Routes>
  );
};

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const { setSystemMetrics, addNotification } = useAppStore();

  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Initialize WebSocket connection
        const wsUrl = import.meta.env.DEV 
          ? 'ws://localhost:8000/ws'
          : 'wss://ws.timpayne.net/ws';

        const connected = await websocketService.connect(wsUrl);
        
        if (connected) {
          setConnectionStatus('connected');
          addNotification({
            id: 'connection-success',
            type: 'success',
            title: 'Connected',
            message: 'Successfully connected to PyGent Factory backend'
          });
        } else {
          setConnectionStatus('disconnected');
          addNotification({
            id: 'connection-failed',
            type: 'error',
            title: 'Connection Failed',
            message: 'Failed to connect to PyGent Factory backend'
          });
        }

        // Set up WebSocket event handlers
        websocketService.on('system_metrics', (metrics) => {
          setSystemMetrics(metrics);
        });

        websocketService.on('connection_status', (status) => {
          setConnectionStatus(status.connected ? 'connected' : 'disconnected');
        });

        // Simulate loading time for smooth UX
        await new Promise(resolve => setTimeout(resolve, 1000));
        
      } catch (error) {
        console.error('Failed to initialize app:', error);
        setConnectionStatus('disconnected');
        addNotification({
          id: 'init-error',
          type: 'error',
          title: 'Initialization Error',
          message: 'Failed to initialize PyGent Factory'
        });
      } finally {
        setIsLoading(false);
      }
    };

    initializeApp();

    // Cleanup on unmount
    return () => {
      websocketService.disconnect();
    };
  }, [setSystemMetrics, addNotification]);

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-background font-sans antialiased">
          <AppLayout connectionStatus={connectionStatus}>
            <ViewRouter />
          </AppLayout>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;