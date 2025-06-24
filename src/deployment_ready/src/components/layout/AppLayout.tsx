import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useAppStore } from '@/stores/appStore';

interface AppLayoutProps {
  children: React.ReactNode;
  connectionStatus: 'connecting' | 'connected' | 'disconnected';
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children, connectionStatus }) => {
  const { ui } = useAppStore();

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <div className={`${ui.sidebarOpen ? 'w-64' : 'w-16'} transition-all duration-300 ease-in-out`}>
        <Sidebar />
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header connectionStatus={connectionStatus} />
        
        <main className="flex-1 overflow-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
};