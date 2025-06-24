import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  MessageSquare,
  Brain,
  Dna,
  Search,
  Activity,
  Package,
  Settings,
  ChevronLeft,
  Bot,
  Server,
  FileSearch,
  BookOpen,
  Zap,
  Cog,
  BarChart3
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useUI, useSystem, useAppStore } from '@/stores/appStore';
import { ViewType } from '@/types';
import { cn } from '@/utils/cn';

// Map ViewType to route paths
const viewTypeToRoute: Record<ViewType, string> = {
  [ViewType.CHAT]: '/chat',
  [ViewType.REASONING]: '/reasoning',
  [ViewType.EVOLUTION]: '/evolution',
  [ViewType.SEARCH]: '/search',
  [ViewType.RESEARCH_ANALYSIS]: '/research-analysis',
  [ViewType.MONITORING]: '/monitoring',
  [ViewType.MCP_MARKETPLACE]: '/mcp-marketplace',
  [ViewType.OLLAMA]: '/ollama',
  [ViewType.STARTUP_DASHBOARD]: '/startup',
  [ViewType.STARTUP_ORCHESTRATION]: '/startup/orchestration',
  [ViewType.STARTUP_CONFIGURATION]: '/startup/configuration',
  [ViewType.STARTUP_MONITORING]: '/startup/monitoring',
  [ViewType.SETTINGS]: '/settings'
};

// Map route paths to ViewType
const routeToViewType: Record<string, ViewType> = {
  '/chat': ViewType.CHAT,
  '/reasoning': ViewType.REASONING,
  '/evolution': ViewType.EVOLUTION,
  '/search': ViewType.SEARCH,
  '/research-analysis': ViewType.RESEARCH_ANALYSIS,
  '/monitoring': ViewType.MONITORING,
  '/mcp-marketplace': ViewType.MCP_MARKETPLACE,
  '/ollama': ViewType.OLLAMA,
  '/startup': ViewType.STARTUP_DASHBOARD,
  '/startup/orchestration': ViewType.STARTUP_ORCHESTRATION,
  '/startup/configuration': ViewType.STARTUP_CONFIGURATION,
  '/startup/monitoring': ViewType.STARTUP_MONITORING,
  '/settings': ViewType.SETTINGS
};

interface NavigationItem {
  id: ViewType;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
  route: string;
  subItems?: NavigationItem[];
}

const navigationItems: NavigationItem[] = [
  {
    id: ViewType.CHAT,
    label: 'Chat',
    icon: MessageSquare,
    description: 'AI Agent Conversations',
    route: '/chat'
  },
  {
    id: ViewType.REASONING,
    label: 'Reasoning',
    icon: Brain,
    description: 'Tree of Thought Analysis',
    route: '/reasoning'
  },
  {
    id: ViewType.EVOLUTION,
    label: 'Evolution',
    icon: Dna,
    description: 'Recipe Optimization',
    route: '/evolution'
  },
  {
    id: ViewType.SEARCH,
    label: 'Search',
    icon: Search,
    description: 'Vector Search & Retrieval',
    route: '/search'
  },
  {
    id: ViewType.RESEARCH_ANALYSIS,
    label: 'Research & Analysis',
    icon: FileSearch,
    description: 'Automated Research Pipeline',
    route: '/research-analysis'
  },
  {
    id: ViewType.MONITORING,
    label: 'Monitoring',
    icon: Activity,
    description: 'System Performance',
    route: '/monitoring'
  },
  {
    id: ViewType.MCP_MARKETPLACE,
    label: 'MCP Servers',
    icon: Package,
    description: 'Model Context Protocol',
    route: '/mcp-marketplace'
  },
  {
    id: ViewType.OLLAMA,
    label: 'Ollama',
    icon: Server,
    description: 'AI Model Management',
    route: '/ollama'
  },
  {
    id: ViewType.STARTUP_DASHBOARD,
    label: 'Startup Service',
    icon: Zap,
    description: 'System Startup & Orchestration',
    route: '/startup',
    subItems: [
      {
        id: ViewType.STARTUP_ORCHESTRATION,
        label: 'Orchestration',
        icon: Cog,
        description: 'Service Orchestration & Sequences',
        route: '/startup/orchestration'
      },
      {
        id: ViewType.STARTUP_CONFIGURATION,
        label: 'Configuration',
        icon: Settings,
        description: 'Service Configuration & Profiles',
        route: '/startup/configuration'
      },
      {
        id: ViewType.STARTUP_MONITORING,
        label: 'Monitoring',
        icon: BarChart3,
        description: 'Real-time Monitoring & Logs',
        route: '/startup/monitoring'
      }
    ]
  },
  {
    id: ViewType.SETTINGS,
    label: 'Settings',
    icon: Settings,
    description: 'Configuration & Preferences',
    route: '/settings'
  }
];

// Documentation link (internal route)
const documentationItem = {
  label: 'Documentation',
  icon: BookOpen,
  description: 'Complete PyGent Factory Docs',
  route: '/docs',
  external: false
};

export const Sidebar: React.FC = () => {
  const { ui, setActiveView, setSidebarOpen } = useUI();
  const { systemMetrics, mcpServers } = useSystem();
  const { startupService } = useAppStore();
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedItems, setExpandedItems] = React.useState<Set<ViewType>>(new Set([ViewType.STARTUP_DASHBOARD]));

  const getStatusIndicator = (viewType: ViewType) => {
    switch (viewType) {
      case ViewType.MONITORING:
        if (systemMetrics) {
          const isHealthy = systemMetrics.cpu.usage_percent < 80 &&
                           systemMetrics.memory.usage_percent < 80;
          return (
            <div className={cn(
              'w-2 h-2 rounded-full',
              isHealthy ? 'bg-green-500' : 'bg-yellow-500'
            )} />
          );
        }
        break;
      case ViewType.MCP_MARKETPLACE:
        const runningServers = mcpServers.filter(s => s.status === 'running').length;
        if (runningServers > 0) {
          return (
            <span className="text-xs bg-primary text-primary-foreground rounded-full px-1.5 py-0.5">
              {runningServers}
            </span>
          );
        }
        break;
      case ViewType.STARTUP_DASHBOARD:
        const runningServices = startupService.services.filter(s => s.status === 'running').length;
        const activeSequences = startupService.activeSequences.length;
        if (runningServices > 0 || activeSequences > 0) {
          return (
            <div className="flex items-center space-x-1">
              {runningServices > 0 && (
                <div className="w-2 h-2 rounded-full bg-green-500" />
              )}
              {activeSequences > 0 && (
                <span className="text-xs bg-blue-500 text-white rounded-full px-1.5 py-0.5">
                  {activeSequences}
                </span>
              )}
            </div>
          );
        }
        break;
    }
    return null;
  };

  return (
    <div className="h-full bg-card border-r flex flex-col">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Bot className="h-6 w-6 text-primary" />
            <h2 className="font-semibold text-lg">PyGent Factory</h2>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(false)}
            className="md:hidden"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          Advanced AI Reasoning System
        </p>
      </div>

      {/* Navigation */}
      <div className="flex-1 p-4 space-y-2">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isActive = ui.activeView === item.id;
          const hasSubItems = item.subItems && item.subItems.length > 0;
          const isExpanded = expandedItems.has(item.id);
          const isSubItemActive = hasSubItems && item.subItems?.some(subItem => ui.activeView === subItem.id);

          return (
            <div key={item.id} className="space-y-1">
              <Button
                variant={isActive || isSubItemActive ? "default" : "ghost"}
                className={cn(
                  "w-full justify-start h-auto p-3",
                  (isActive || isSubItemActive) && "bg-primary text-primary-foreground"
                )}
                onClick={() => {
                  if (hasSubItems) {
                    // Toggle expansion for items with sub-items
                    const newExpanded = new Set(expandedItems);
                    if (isExpanded) {
                      newExpanded.delete(item.id);
                    } else {
                      newExpanded.add(item.id);
                    }
                    setExpandedItems(newExpanded);
                  } else {
                    // Navigate for items without sub-items
                    setActiveView(item.id);
                    navigate(item.route);
                    if (window.innerWidth < 768) {
                      setSidebarOpen(false);
                    }
                  }
                }}
              >
                <div className="flex items-center space-x-3 w-full">
                  <Icon className="h-5 w-5 flex-shrink-0" />
                  <div className="flex-1 text-left">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{item.label}</span>
                      <div className="flex items-center space-x-2">
                        {getStatusIndicator(item.id)}
                        {hasSubItems && (
                          <ChevronLeft className={cn(
                            "h-4 w-4 transition-transform",
                            isExpanded && "rotate-90"
                          )} />
                        )}
                      </div>
                    </div>
                    <p className="text-xs opacity-70 mt-0.5">
                      {item.description}
                    </p>
                  </div>
                </div>
              </Button>

              {/* Sub-items */}
              {hasSubItems && isExpanded && (
                <div className="ml-6 space-y-1">
                  {item.subItems?.map((subItem) => {
                    const SubIcon = subItem.icon;
                    const isSubActive = ui.activeView === subItem.id;

                    return (
                      <Button
                        key={subItem.id}
                        variant={isSubActive ? "default" : "ghost"}
                        size="sm"
                        className={cn(
                          "w-full justify-start h-auto p-2",
                          isSubActive && "bg-primary text-primary-foreground"
                        )}
                        onClick={() => {
                          setActiveView(subItem.id);
                          navigate(subItem.route);
                          if (window.innerWidth < 768) {
                            setSidebarOpen(false);
                          }
                        }}
                      >
                        <div className="flex items-center space-x-2 w-full">
                          <SubIcon className="h-4 w-4 flex-shrink-0" />
                          <div className="flex-1 text-left">
                            <span className="text-sm font-medium">{subItem.label}</span>
                            <p className="text-xs opacity-70 mt-0.5">
                              {subItem.description}
                            </p>
                          </div>
                        </div>
                      </Button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}

        {/* Documentation Link */}
        <div className="pt-2 border-t">
          <Button
            variant="ghost"
            className={`w-full justify-start h-auto p-3 text-blue-600 hover:text-blue-700 hover:bg-blue-50 ${
              ui.activeView === 'docs' ? 'bg-blue-50 text-blue-700' : ''
            }`}
            onClick={() => {
              setActiveView('docs' as ViewType);
              navigate(documentationItem.route);
              if (window.innerWidth < 768) {
                setSidebarOpen(false);
              }
            }}
          >
            <div className="flex items-center space-x-3 w-full">
              <BookOpen className="h-5 w-5 flex-shrink-0" />
              <div className="flex-1 text-left">
                <div className="flex items-center justify-between">
                  <span className="font-medium">{documentationItem.label}</span>
                  <span className="text-xs opacity-70">📖</span>
                </div>
                <p className="text-xs opacity-70 mt-0.5">
                  {documentationItem.description}
                </p>
              </div>
            </div>
          </Button>
        </div>
      </div>

      {/* System Status */}
      <div className="p-4 border-t">
        <Card>
          <CardContent className="p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">System Status</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full" />
                <span className="text-xs text-muted-foreground">Online</span>
              </div>
            </div>
            
            {systemMetrics && (
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">CPU</span>
                  <span>{systemMetrics.cpu.usage_percent.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Memory</span>
                  <span>{systemMetrics.memory.usage_percent.toFixed(1)}%</span>
                </div>
                {systemMetrics.gpu && (
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">GPU</span>
                    <span>{systemMetrics.gpu.usage_percent.toFixed(1)}%</span>
                  </div>
                )}
              </div>
            )}
            
            <div className="flex justify-between text-xs mt-2">
              <span className="text-muted-foreground">MCP Servers</span>
              <span>{mcpServers.filter(s => s.status === 'running').length}/{mcpServers.length}</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
