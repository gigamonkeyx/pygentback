/**
 * Startup Monitoring Page
 * Real-time monitoring, logs, and metrics for startup services
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart3, 
  Activity, 
  FileText, 
  Download,
  Filter,
  RefreshCw,
  Trash2,
  Search,
  AlertTriangle,
  Info,
  XCircle,
  CheckCircle
} from 'lucide-react';
import { useStartupService } from '@/stores/appStore';
import { useLogs, useMetrics } from '@/hooks/useStartupService';
import { LogViewer, SystemMetricsPanel } from '@/components/startup';

const StartupMonitoringPage: React.FC = () => {
  const { startupService, clearStartupLogs } = useStartupService();
  const { data: logs, isLoading: logsLoading, refetch: refetchLogs } = useLogs();
  const { data: metrics, isLoading: metricsLoading, refetch: refetchMetrics } = useMetrics();
  const [logFilter, setLogFilter] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Auto-refresh logs and metrics
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      refetchLogs();
      refetchMetrics();
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh, refetchLogs, refetchMetrics]);

  const getLogIcon = (level: string) => {
    switch (level.toLowerCase()) {
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'info':
        return <Info className="h-4 w-4 text-blue-500" />;
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      default:
        return <FileText className="h-4 w-4 text-gray-500" />;
    }
  };

  const getLogColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'error':
        return 'border-l-red-500 bg-red-50';
      case 'warning':
        return 'border-l-yellow-500 bg-yellow-50';
      case 'info':
        return 'border-l-blue-500 bg-blue-50';
      case 'success':
        return 'border-l-green-500 bg-green-50';
      default:
        return 'border-l-gray-500 bg-gray-50';
    }
  };

  const filteredLogs = startupService.logs.filter(log => {
    if (logFilter === 'all') return true;
    return log.data.level === logFilter;
  });

  const logLevelCounts = startupService.logs.reduce((acc, log) => {
    acc[log.data.level] = (acc[log.data.level] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="h-full overflow-auto">
      <div className="container mx-auto p-6 space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center">
              <BarChart3 className="h-8 w-8 mr-3 text-green-500" />
              Startup Monitoring
            </h1>
            <p className="text-muted-foreground mt-2">
              Real-time monitoring, logs, and performance metrics
            </p>
          </div>
          
          <div className="flex space-x-2">
            <Button
              variant={autoRefresh ? "default" : "outline"}
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
              Auto Refresh
            </Button>
            <Button variant="outline" onClick={() => {
              refetchLogs();
              refetchMetrics();
            }}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh Now
            </Button>
          </div>
        </div>

        {/* Monitoring Tabs */}
        <Tabs defaultValue="logs" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="logs">System Logs</TabsTrigger>
            <TabsTrigger value="metrics">Performance Metrics</TabsTrigger>
            <TabsTrigger value="alerts">Alerts & Events</TabsTrigger>
          </TabsList>

          {/* System Logs Tab */}
          <TabsContent value="logs" className="space-y-6">
            <LogViewer
              logs={startupService.logs}
              onClear={() => clearStartupLogs()}
              onFilter={(filters) => {
                console.log('Apply log filters:', filters);
              }}
              onExport={() => {
                console.log('Export logs');
              }}
              maxLines={500}
              autoScroll={autoRefresh}
            />
          </TabsContent>

          {/* Performance Metrics Tab */}
          <TabsContent value="metrics" className="space-y-6">
            <SystemMetricsPanel
              metrics={metrics || {}}
              realtimeData={startupService.realtimeMetrics}
              onMetricSelect={(metric) => {
                console.log('Selected metric:', metric);
              }}
              refreshInterval={autoRefresh ? 5000 : 0}
            />
          </TabsContent>

          {/* Alerts & Events Tab */}
          <TabsContent value="alerts" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Alerts & Events</CardTitle>
                <CardDescription>
                  System alerts, warnings, and important events
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <AlertTriangle className="h-16 w-16 mx-auto mb-4 text-gray-400" />
                  <h3 className="text-lg font-semibold mb-2">No Active Alerts</h3>
                  <p className="text-gray-600">
                    System alerts and critical events will appear here.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default StartupMonitoringPage;
