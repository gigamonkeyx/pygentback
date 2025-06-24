import React from 'react';
import { Bell, Wifi, WifiOff, Loader2 } from 'lucide-react';
import { useAppStore } from '@/stores/appStore';

interface HeaderProps {
  connectionStatus: 'connecting' | 'connected' | 'disconnected';
}

export const Header: React.FC<HeaderProps> = ({ connectionStatus }) => {
  const { ui, notifications } = useAppStore();
  
  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Wifi className="w-4 h-4 text-green-500" />;
      case 'connecting':
        return <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />;
      case 'disconnected':
        return <WifiOff className="w-4 h-4 text-red-500" />;
    }
  };

  const getConnectionText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected to PyGent Factory';
      case 'connecting':
        return 'Connecting...';
      case 'disconnected':
        return 'Disconnected';
    }
  };

  const unreadNotifications = notifications.filter(n => !n.read).length;

  return (
    <header className="h-16 bg-card border-b border-border flex items-center justify-between px-6">
      <div className="flex items-center space-x-4">
        <h1 className="text-xl font-semibold text-foreground">
          {ui.activeView === 'chat' && 'Multi-Agent Chat'}
          {ui.activeView === 'reasoning' && 'Tree of Thought Reasoning'}
          {ui.activeView === 'monitoring' && 'System Monitoring'}
          {ui.activeView === 'mcp-marketplace' && 'MCP Marketplace'}
          {ui.activeView === 'settings' && 'Settings'}
        </h1>
      </div>

      <div className="flex items-center space-x-4">
        {/* Connection Status */}
        <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-muted">
          {getConnectionIcon()}
          <span className="text-sm text-muted-foreground">
            {getConnectionText()}
          </span>
        </div>

        {/* Notifications */}
        <div className="relative">
          <button className="p-2 rounded-lg hover:bg-accent transition-colors">
            <Bell className="w-5 h-5 text-muted-foreground" />
            {unreadNotifications > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                {unreadNotifications > 9 ? '9+' : unreadNotifications}
              </span>
            )}
          </button>
        </div>

        {/* User Menu */}
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
            <span className="text-sm font-medium text-primary-foreground">U</span>
          </div>
        </div>
      </div>
    </header>
  );
};