import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  MessageSquare, 
  Brain, 
  Activity, 
  Package, 
  Settings,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { useAppStore } from '@/stores/appStore';

const navigation = [
  { name: 'Chat', href: '/chat', icon: MessageSquare },
  { name: 'Reasoning', href: '/reasoning', icon: Brain },
  { name: 'Monitoring', href: '/monitoring', icon: Activity },
  { name: 'MCP Marketplace', href: '/mcp-marketplace', icon: Package },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  const { ui, toggleSidebar } = useAppStore();

  return (
    <div className="h-full bg-card border-r border-border flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          {ui.sidebarOpen && (
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Brain className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="font-semibold text-foreground">PyGent Factory</span>
            </div>
          )}
          
          <button
            onClick={toggleSidebar}
            className="p-1 rounded-md hover:bg-accent transition-colors"
          >
            {ui.sidebarOpen ? (
              <ChevronLeft className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              `flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent'
              }`
            }
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            {ui.sidebarOpen && <span className="font-medium">{item.name}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      {ui.sidebarOpen && (
        <div className="p-4 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div>PyGent Factory v1.0.0</div>
            <div>Advanced AI Reasoning</div>
          </div>
        </div>
      )}
    </div>
  );
};