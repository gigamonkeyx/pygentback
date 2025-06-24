import React, { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { Bot, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuth } from '@/stores/appStore';
import { cn } from '@/utils/cn';

export const LoginPage: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const { isAuthenticated, setUser, setAuthenticated } = useAuth();

  // Redirect if already authenticated
  if (isAuthenticated) {
    return <Navigate to="/chat" replace />;
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // For demo purposes, accept any non-empty credentials
      if (username.trim() && password.trim()) {
        const user = {
          id: '1',
          username: username,
          email: `${username}@example.com`,
          role: 'developer' as const,
          permissions: [
            'chat_access',
            'reasoning_control',
            'evolution_control',
            'search_management',
            'mcp_management',
            'system_monitoring'
          ] as any[],
          preferences: {
            theme: 'system' as const,
            language: 'en',
            timezone: 'UTC',
            notifications: {
              email: true,
              push: true,
              system_alerts: true,
              ai_updates: true,
              mcp_events: true
            },
            dashboard: {
              default_view: 'chat' as const,
              sidebar_collapsed: false,
              auto_refresh: true,
              refresh_interval: 30,
              chart_animations: true
            }
          },
          created_at: new Date(),
          last_login: new Date()
        };

        setUser(user);
        setAuthenticated(true);
      } else {
        setError('Please enter both username and password');
      }
    } catch (err) {
      setError('Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className="p-3 bg-primary/10 rounded-full">
              <Bot className="h-8 w-8 text-primary" />
            </div>
          </div>
          <CardTitle className="text-2xl font-bold">Welcome to PyGent Factory</CardTitle>
          <CardDescription>
            Advanced AI Reasoning System - Sign in to continue
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleLogin} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="username" className="text-sm font-medium">
                Username
              </label>
              <Input
                id="username"
                type="text"
                placeholder="Enter your username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={isLoading}
                required
              />
            </div>
            
            <div className="space-y-2">
              <label htmlFor="password" className="text-sm font-medium">
                Password
              </label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoading}
                  required
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                  onClick={() => setShowPassword(!showPassword)}
                  disabled={isLoading}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>

            {error && (
              <div className="text-sm text-red-500 bg-red-50 dark:bg-red-900/20 p-3 rounded-md">
                {error}
              </div>
            )}

            <Button
              type="submit"
              className="w-full"
              disabled={isLoading}
            >
              {isLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="spinner" />
                  <span>Signing in...</span>
                </div>
              ) : (
                'Sign In'
              )}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-muted-foreground">
              Demo credentials: Any username and password
            </p>
          </div>

          <div className="mt-6 pt-6 border-t">
            <div className="text-center space-y-2">
              <h3 className="text-sm font-medium">System Features</h3>
              <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                <div>• Tree of Thought Reasoning</div>
                <div>• Recipe Evolution</div>
                <div>• Vector Search</div>
                <div>• MCP Integration</div>
                <div>• Real-time Monitoring</div>
                <div>• Multi-Agent Chat</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
