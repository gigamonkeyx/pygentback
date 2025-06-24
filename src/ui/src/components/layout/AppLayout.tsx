import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useUI, useAuth } from '@/stores/appStore';
import { ViewType } from '@/types';
import { cn } from '@/utils/cn';

interface AppLayoutProps {
  children: React.ReactNode;
}

// Map route paths to ViewType
const routeToViewType: Record<string, ViewType> = {
  '/chat': ViewType.CHAT,
  '/reasoning': ViewType.REASONING,
  '/evolution': ViewType.EVOLUTION,
  '/search': ViewType.SEARCH,
  '/monitoring': ViewType.MONITORING,
  '/mcp-marketplace': ViewType.MCP_MARKETPLACE,
  '/ollama': ViewType.OLLAMA,
  '/startup': ViewType.STARTUP_DASHBOARD,
  '/startup/orchestration': ViewType.STARTUP_ORCHESTRATION,
  '/startup/configuration': ViewType.STARTUP_CONFIGURATION,
  '/startup/monitoring': ViewType.STARTUP_MONITORING,
  '/settings': ViewType.SETTINGS
};

// Get page title based on ViewType
const getPageTitle = (viewType: ViewType): string => {
  switch (viewType) {
    case ViewType.CHAT:
      return 'AI Chat Interface';
    case ViewType.REASONING:
      return 'Reasoning Engine';
    case ViewType.EVOLUTION:
      return 'Evolution System';
    case ViewType.SEARCH:
      return 'Search & Discovery';
    case ViewType.MONITORING:
      return 'System Monitoring';
    case ViewType.MCP_MARKETPLACE:
      return 'MCP Marketplace';
    case ViewType.OLLAMA:
      return 'Ollama Management';
    case ViewType.STARTUP_DASHBOARD:
      return 'Startup Service Dashboard';
    case ViewType.STARTUP_ORCHESTRATION:
      return 'Service Orchestration';
    case ViewType.STARTUP_CONFIGURATION:
      return 'Configuration Management';
    case ViewType.STARTUP_MONITORING:
      return 'Startup Monitoring';
    case ViewType.SETTINGS:
      return 'Settings';
    default:
      return 'PyGent Factory';
  }
};

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const { ui, setSidebarOpen, setActiveView } = useUI();
  const { isAuthenticated } = useAuth();
  const location = useLocation();

  // Debug logging
  console.log('AppLayout Debug:', {
    isAuthenticated,
    sidebarOpen: ui.sidebarOpen,
    activeView: ui.activeView
  });

  useEffect(() => {
    // Note: WebSocket connection is handled in App.tsx
    // We don't need to reconnect here as it's already a singleton
  }, [isAuthenticated]);

  useEffect(() => {
    // Synchronize activeView with current URL
    const currentViewType = routeToViewType[location.pathname];
    if (currentViewType && currentViewType !== ui.activeView) {
      setActiveView(currentViewType);
    }
  }, [location.pathname, ui.activeView, setActiveView]);

  // Update document title based on active view
  useEffect(() => {
    const pageTitle = getPageTitle(ui.activeView);
    document.title = `${pageTitle} - PyGent Factory`;
  }, [ui.activeView]);

  useEffect(() => {
    // Handle responsive sidebar behavior
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [setSidebarOpen]);

  if (!isAuthenticated) {
    return <>{children}</>;
  }

  return (
    <div className="h-screen flex bg-background">
      {/* Sidebar */}
      <div className="w-64 h-full bg-card border-r flex-shrink-0">
        <Sidebar />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 h-screen">
        <Header />
        <main className="flex-1 overflow-hidden h-full">
          {children}
        </main>
      </div>
    </div>
  );
};
