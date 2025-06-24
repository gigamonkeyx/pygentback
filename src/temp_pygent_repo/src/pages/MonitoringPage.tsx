import React, { useEffect } from 'react';
import { Activity, Cpu, HardDrive, Wifi, Zap } from 'lucide-react';
import { useAppStore } from '@/stores/appStore';
import { websocketService } from '@/services/websocket';

export const MonitoringPage: React.FC = () => {
  const { systemMetrics } = useAppStore();

  useEffect(() => {
    // Request initial metrics
    websocketService.requestSystemMetrics();

    // Set up periodic metrics requests
    const interval = setInterval(() => {
      websocketService.requestSystemMetrics();
    }, 5000); // Every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'text-green-500';
      case 'offline':
        return 'text-red-500';
      case 'error':
        return 'text-yellow-500';
      default:
        return 'text-muted-foreground';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return '🟢';
      case 'offline':
        return '🔴';
      case 'error':
        return '🟡';
      default:
        return '⚪';
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-3">
        <Activity className="w-8 h-8 text-primary" />
        <div>
          <h1 className="text-2xl font-bold text-foreground">System Monitoring</h1>
          <p className="text-muted-foreground">Real-time system and agent performance</p>
        </div>
      </div>

      {/* System Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* CPU Usage */}
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <Cpu className="w-6 h-6 text-primary" />
            <div>
              <p className="text-sm font-medium text-foreground">CPU Usage</p>
              <p className="text-2xl font-bold text-foreground">
                {systemMetrics?.cpu ? `${systemMetrics.cpu.toFixed(1)}%` : 'N/A'}
              </p>
            </div>
          </div>
          {systemMetrics?.cpu && (
            <div className="mt-4">
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    systemMetrics.cpu > 80 ? 'bg-red-500' :
                    systemMetrics.cpu > 60 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${Math.min(systemMetrics.cpu, 100)}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Memory Usage */}
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <HardDrive className="w-6 h-6 text-primary" />
            <div>
              <p className="text-sm font-medium text-foreground">Memory Usage</p>
              <p className="text-2xl font-bold text-foreground">
                {systemMetrics?.memory ? `${systemMetrics.memory.toFixed(1)}%` : 'N/A'}
              </p>
            </div>
          </div>
          {systemMetrics?.memory && (
            <div className="mt-4">
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    systemMetrics.memory > 80 ? 'bg-red-500' :
                    systemMetrics.memory > 60 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${Math.min(systemMetrics.memory, 100)}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* GPU Usage */}
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <Zap className="w-6 h-6 text-primary" />
            <div>
              <p className="text-sm font-medium text-foreground">GPU Usage</p>
              <p className="text-2xl font-bold text-foreground">
                {systemMetrics?.gpu ? `${systemMetrics.gpu.toFixed(1)}%` : 'N/A'}
              </p>
            </div>
          </div>
          {systemMetrics?.gpu && (
            <div className="mt-4">
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    systemMetrics.gpu > 80 ? 'bg-red-500' :
                    systemMetrics.gpu > 60 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${Math.min(systemMetrics.gpu, 100)}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Network */}
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <Wifi className="w-6 h-6 text-primary" />
            <div>
              <p className="text-sm font-medium text-foreground">Network</p>
              <div className="text-sm text-muted-foreground">
                <p>↑ {systemMetrics?.network ? formatBytes(systemMetrics.network.upload) : 'N/A'}</p>
                <p>↓ {systemMetrics?.network ? formatBytes(systemMetrics.network.download) : 'N/A'}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Agent Status */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">Agent Services</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* ToT Reasoning Agent */}
          <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
            <div className="flex items-center space-x-3">
              <span className="text-2xl">
                {getStatusIcon(systemMetrics?.agents?.tot_reasoning || 'offline')}
              </span>
              <div>
                <p className="font-medium text-foreground">ToT Reasoning Agent</p>
                <p className="text-sm text-muted-foreground">Port 8001</p>
              </div>
            </div>
            <span className={`text-sm font-medium ${getStatusColor(systemMetrics?.agents?.tot_reasoning || 'offline')}`}>
              {systemMetrics?.agents?.tot_reasoning || 'offline'}
            </span>
          </div>

          {/* RAG Retrieval Agent */}
          <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
            <div className="flex items-center space-x-3">
              <span className="text-2xl">
                {getStatusIcon(systemMetrics?.agents?.rag_retrieval || 'offline')}
              </span>
              <div>
                <p className="font-medium text-foreground">RAG Retrieval Agent</p>
                <p className="text-sm text-muted-foreground">Port 8002</p>
              </div>
            </div>
            <span className={`text-sm font-medium ${getStatusColor(systemMetrics?.agents?.rag_retrieval || 'offline')}`}>
              {systemMetrics?.agents?.rag_retrieval || 'offline'}
            </span>
          </div>
        </div>
      </div>

      {/* System Information */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">System Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Backend API</p>
            <p className="font-medium text-foreground">
              {import.meta.env.DEV ? 'http://localhost:8000' : 'https://api.timpayne.net'}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">WebSocket</p>
            <p className="font-medium text-foreground">
              {import.meta.env.DEV ? 'ws://localhost:8000/ws' : 'wss://ws.timpayne.net/ws'}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Environment</p>
            <p className="font-medium text-foreground">
              {import.meta.env.DEV ? 'Development' : 'Production'}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Version</p>
            <p className="font-medium text-foreground">PyGent Factory v1.0.0</p>
          </div>
        </div>
      </div>
    </div>
  );
};