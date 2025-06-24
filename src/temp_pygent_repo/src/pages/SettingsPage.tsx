import React from 'react';
import { Settings, Moon, Sun, Monitor, Bell, Database, Wifi } from 'lucide-react';
import { useAppStore } from '@/stores/appStore';

export const SettingsPage: React.FC = () => {
  const { ui, setTheme } = useAppStore();

  const themes = [
    { id: 'light', name: 'Light', icon: Sun },
    { id: 'dark', name: 'Dark', icon: Moon },
    { id: 'system', name: 'System', icon: Monitor }
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-3">
        <Settings className="w-8 h-8 text-primary" />
        <div>
          <h1 className="text-2xl font-bold text-foreground">Settings</h1>
          <p className="text-muted-foreground">Configure your PyGent Factory experience</p>
        </div>
      </div>

      {/* Appearance */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">Appearance</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Theme
            </label>
            <div className="grid grid-cols-3 gap-3">
              {themes.map((theme) => (
                <button
                  key={theme.id}
                  onClick={() => setTheme(theme.id as any)}
                  className={`flex items-center space-x-2 p-3 rounded-lg border transition-colors ${
                    ui.theme === theme.id
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-border bg-background text-foreground hover:bg-accent'
                  }`}
                >
                  <theme.icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{theme.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Notifications */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">Notifications</h2>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Bell className="w-5 h-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium text-foreground">System Alerts</p>
                <p className="text-xs text-muted-foreground">Get notified about system issues</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-muted peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
            </label>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Bell className="w-5 h-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium text-foreground">Agent Responses</p>
                <p className="text-xs text-muted-foreground">Get notified when agents respond</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-muted peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
            </label>
          </div>
        </div>
      </div>

      {/* System Configuration */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">System Configuration</h2>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                API Endpoint
              </label>
              <div className="flex items-center space-x-2 p-2 bg-muted rounded-lg">
                <Wifi className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm text-foreground">
                  {import.meta.env.DEV ? 'http://localhost:8000' : 'https://api.timpayne.net'}
                </span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                WebSocket URL
              </label>
              <div className="flex items-center space-x-2 p-2 bg-muted rounded-lg">
                <Wifi className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm text-foreground">
                  {import.meta.env.DEV ? 'ws://localhost:8000/ws' : 'wss://ws.timpayne.net/ws'}
                </span>
              </div>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Database Connection
            </label>
            <div className="flex items-center space-x-2 p-2 bg-muted rounded-lg">
              <Database className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-foreground">PostgreSQL (localhost:5432)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Agent Configuration */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">Agent Configuration</h2>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Default Reasoning Mode
              </label>
              <select className="w-full p-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:border-transparent">
                <option value="adaptive">Adaptive</option>
                <option value="breadth_first">Breadth First</option>
                <option value="depth_first">Depth First</option>
                <option value="best_first">Best First</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Response Timeout (seconds)
              </label>
              <input
                type="number"
                defaultValue={30}
                min={5}
                max={300}
                className="w-full p-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-foreground">Enable GPU Acceleration</p>
              <p className="text-xs text-muted-foreground">Use GPU for faster AI processing when available</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-muted peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
            </label>
          </div>
        </div>
      </div>

      {/* About */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">About</h2>
        
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Version</span>
            <span className="text-foreground">PyGent Factory v1.0.0</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Environment</span>
            <span className="text-foreground">
              {import.meta.env.DEV ? 'Development' : 'Production'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Build Date</span>
            <span className="text-foreground">{new Date().toLocaleDateString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
};