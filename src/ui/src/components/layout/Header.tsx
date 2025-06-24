import React from 'react';
import {
  Menu,
  Bell,
  User,
  Sun,
  Moon,
  Monitor,
  LogOut,
  Settings,
  ChevronRight,
  Zap,
  Cog,
  BarChart3
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useUI, useAuth } from '@/stores/appStore';
import { cn } from '@/utils/cn';
import { ViewType } from '@/types';

export const Header: React.FC = () => {
  const { ui, setSidebarOpen, setTheme, addNotification } = useUI();
  const { user, setAuthenticated } = useAuth();

  const handleThemeChange = (theme: 'light' | 'dark' | 'system') => {
    setTheme(theme);
    // Apply theme to document
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    
    if (theme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      root.classList.add(systemTheme);
    } else {
      root.classList.add(theme);
    }
  };

  const handleLogout = () => {
    setAuthenticated(false);
    addNotification({
      type: 'info',
      title: 'Logged Out',
      message: 'You have been successfully logged out.'
    });
  };

  const getThemeIcon = () => {
    switch (ui.theme) {
      case 'light':
        return <Sun className="h-4 w-4" />;
      case 'dark':
        return <Moon className="h-4 w-4" />;
      default:
        return <Monitor className="h-4 w-4" />;
    }
  };

  const unreadNotifications = ui.notifications.filter(n => !n.read).length;

  // Generate breadcrumb for startup service views
  const getBreadcrumb = () => {
    const breadcrumbs = [];

    if (ui.activeView.startsWith('startup_')) {
      breadcrumbs.push({
        label: 'Startup Service',
        icon: Zap,
        href: '/startup'
      });

      switch (ui.activeView) {
        case ViewType.STARTUP_ORCHESTRATION:
          breadcrumbs.push({
            label: 'Orchestration',
            icon: Cog,
            href: '/startup/orchestration'
          });
          break;
        case ViewType.STARTUP_CONFIGURATION:
          breadcrumbs.push({
            label: 'Configuration',
            icon: Settings,
            href: '/startup/configuration'
          });
          break;
        case ViewType.STARTUP_MONITORING:
          breadcrumbs.push({
            label: 'Monitoring',
            icon: BarChart3,
            href: '/startup/monitoring'
          });
          break;
      }
    }

    return breadcrumbs;
  };

  const breadcrumbs = getBreadcrumb();

  return (
    <header className="h-16 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex items-center justify-between h-full px-4">
        {/* Left side */}
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(!ui.sidebarOpen)}
          >
            <Menu className="h-5 w-5" />
          </Button>
          
          <div className="hidden md:block">
            {breadcrumbs.length > 0 ? (
              <nav className="flex items-center space-x-2">
                {breadcrumbs.map((crumb, index) => {
                  const Icon = crumb.icon;
                  const isLast = index === breadcrumbs.length - 1;

                  return (
                    <div key={crumb.href} className="flex items-center space-x-2">
                      <div className="flex items-center space-x-1">
                        <Icon className="h-4 w-4 text-muted-foreground" />
                        <span className={cn(
                          "text-sm",
                          isLast ? "font-semibold text-foreground" : "text-muted-foreground"
                        )}>
                          {crumb.label}
                        </span>
                      </div>
                      {!isLast && (
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      )}
                    </div>
                  );
                })}
              </nav>
            ) : (
              <h1 className="text-lg font-semibold">
                {ui.activeView.charAt(0).toUpperCase() + ui.activeView.slice(1).replace('_', ' ')}
              </h1>
            )}
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-2">
          {/* Theme Toggle */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                {getThemeIcon()}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Theme</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => handleThemeChange('light')}>
                <Sun className="h-4 w-4 mr-2" />
                Light
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleThemeChange('dark')}>
                <Moon className="h-4 w-4 mr-2" />
                Dark
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleThemeChange('system')}>
                <Monitor className="h-4 w-4 mr-2" />
                System
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Notifications */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="relative">
                <Bell className="h-5 w-5" />
                {unreadNotifications > 0 && (
                  <span className="absolute -top-1 -right-1 h-5 w-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                    {unreadNotifications > 9 ? '9+' : unreadNotifications}
                  </span>
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-80">
              <DropdownMenuLabel>Notifications</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {ui.notifications.length === 0 ? (
                <div className="p-4 text-center text-muted-foreground">
                  No notifications
                </div>
              ) : (
                <div className="max-h-96 overflow-y-auto">
                  {ui.notifications.slice(0, 5).map((notification) => (
                    <DropdownMenuItem
                      key={notification.id}
                      className="flex flex-col items-start space-y-1 p-3"
                    >
                      <div className="flex items-center justify-between w-full">
                        <span className="font-medium text-sm">
                          {notification.title}
                        </span>
                        <div className={cn(
                          'w-2 h-2 rounded-full',
                          notification.read ? 'bg-gray-300' : 'bg-blue-500'
                        )} />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {notification.message}
                      </p>
                    </DropdownMenuItem>
                  ))}
                </div>
              )}
            </DropdownMenuContent>
          </DropdownMenu>

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="flex items-center space-x-2 px-3">
                <User className="h-5 w-5" />
                <span className="hidden md:inline-block">
                  {user?.username || 'User'}
                </span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium">{user?.username || 'User'}</p>
                  <p className="text-xs text-muted-foreground">{user?.email || 'user@example.com'}</p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleLogout}>
                <LogOut className="h-4 w-4 mr-2" />
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
};
