/**
 * Startup Dashboard Page
 * Main dashboard for the PyGent Factory Startup Service
 */

import React, { useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Zap, 
  Activity, 
  Settings, 
  Play, 
  Square,
  RefreshCw,
  AlertTriangle
} from 'lucide-react';
import { useStartupService } from '@/stores/appStore';
import { ServiceStatusDashboard } from '@/components/startup';
import { ServiceStatus, ServiceType } from '@/types';

const StartupDashboardPage: React.FC = () => {
  const { 
    startupService, 
    updateStartupServices, 
    updateSystemStatus,
    addStartupLog 
  } = useStartupService();

  // Mock data for testing
  useEffect(() => {
    // Initialize with mock data for testing
    const mockServices = [
      {
        service_name: 'PostgreSQL',
        status: ServiceStatus.RUNNING,
        health_score: 0.95,
        last_check: new Date(),
        uptime_seconds: 3600,
        metrics: {
          connections: 12,
          cpu_usage: 15.2,
          memory_usage: 256
        }
      },
      {
        service_name: 'Redis',
        status: ServiceStatus.RUNNING,
        health_score: 0.88,
        last_check: new Date(),
        uptime_seconds: 3500,
        metrics: {
          memory_usage: 128,
          connected_clients: 5,
          operations_per_sec: 1250
        }
      },
      {
        service_name: 'Ollama',
        status: ServiceStatus.STOPPED,
        health_score: 0.0,
        last_check: new Date(),
        uptime_seconds: 0,
        error_message: 'Service not started',
        metrics: {
          models_loaded: 0,
          gpu_usage: 0
        }
      },
      {
        service_name: 'Agent Orchestrator',
        status: ServiceStatus.STARTING,
        health_score: 0.65,
        last_check: new Date(),
        uptime_seconds: 120,
        metrics: {
          active_agents: 0,
          pending_tasks: 3
        }
      }
    ];

    const mockSystemStatus = {
      overall_status: 'healthy' as any,
      health_score: 0.82,
      services: mockServices,
      active_sequences: ['startup-sequence-1'],
      system_metrics: {
        cpu_usage: 45.2,
        memory_usage: 68.5,
        disk_usage: 32.1
      },
      last_updated: new Date()
    };

    updateStartupServices(mockServices);
    updateSystemStatus(mockSystemStatus);

    // Add some mock logs
    addStartupLog({
      type: 'startup_log',
      data: {
        level: 'info',
        logger: 'startup-service',
        message: 'PostgreSQL service started successfully',
        service: 'PostgreSQL',
        details: {}
      },
      timestamp: new Date(),
      source: 'startup-service'
    });

    addStartupLog({
      type: 'startup_log',
      data: {
        level: 'warning',
        logger: 'startup-service',
        message: 'Ollama service failed to start - GPU not available',
        service: 'Ollama',
        details: { error_code: 'GPU_NOT_FOUND' }
      },
      timestamp: new Date(),
      source: 'startup-service'
    });
  }, [updateStartupServices, updateSystemStatus, addStartupLog]);

  const handleServiceAction = (serviceName: string, action: 'start' | 'stop' | 'restart') => {
    console.log(`${action} action triggered for service: ${serviceName}`);
    
    // Mock action feedback
    addStartupLog({
      type: 'startup_log',
      data: {
        level: 'info',
        logger: 'startup-service',
        message: `${action} action initiated for ${serviceName}`,
        service: serviceName,
        details: { action }
      },
      timestamp: new Date(),
      source: 'startup-service'
    });

    // Simulate status change
    setTimeout(() => {
      const newStatus = action === 'start' ? ServiceStatus.STARTING : 
                       action === 'stop' ? ServiceStatus.STOPPING : 
                       ServiceStatus.STARTING;
      
      updateServiceStatus(serviceName, { status: newStatus });
    }, 100);
  };

  const handleRefresh = async () => {
    console.log('Refreshing startup service data...');
    
    // Simulate refresh delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    addStartupLog({
      type: 'startup_log',
      data: {
        level: 'info',
        logger: 'startup-service',
        message: 'Service status refreshed',
        details: { refresh_time: new Date().toISOString() }
      },
      timestamp: new Date(),
      source: 'startup-service'
    });
  };

  return (
    <div className="h-full overflow-auto">
      <div className="container mx-auto p-6 space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center">
              <Zap className="h-8 w-8 mr-3 text-blue-500" />
              Startup Service Dashboard
            </h1>
            <p className="text-muted-foreground mt-2">
              Monitor and control PyGent Factory system startup and service orchestration
            </p>
          </div>
          
          <div className="flex space-x-2">
            <Button variant="outline" onClick={handleRefresh}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button variant="outline">
              <Settings className="h-4 w-4 mr-2" />
              Configure
            </Button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <Activity className="h-8 w-8 text-green-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-muted-foreground">Running Services</p>
                  <p className="text-2xl font-bold">
                    {startupService.services.filter(s => s.status === ServiceStatus.RUNNING).length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <Play className="h-8 w-8 text-blue-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-muted-foreground">Active Sequences</p>
                  <p className="text-2xl font-bold">{startupService.activeSequences.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <Settings className="h-8 w-8 text-purple-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-muted-foreground">Configurations</p>
                  <p className="text-2xl font-bold">{startupService.configurations.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                {startupService.systemStatus?.overall_status === 'healthy' ? (
                  <Activity className="h-8 w-8 text-green-500" />
                ) : (
                  <AlertTriangle className="h-8 w-8 text-yellow-500" />
                )}
                <div className="ml-4">
                  <p className="text-sm font-medium text-muted-foreground">System Health</p>
                  <p className="text-2xl font-bold">
                    {startupService.systemStatus ? 
                      Math.round(startupService.systemStatus.health_score * 100) : 0}%
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Service Status Dashboard */}
        <ServiceStatusDashboard
          services={startupService.services}
          systemStatus={startupService.systemStatus}
          onServiceAction={handleServiceAction}
          onRefresh={handleRefresh}
        />

        {/* Recent Logs */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>
              Latest startup service logs and events
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {startupService.logs.slice(-5).reverse().map((log, index) => (
                <div key={index} className="flex items-center space-x-3 p-2 rounded border">
                  <Badge variant={
                    log.data.level === 'error' ? 'destructive' :
                    log.data.level === 'warning' ? 'secondary' : 'default'
                  }>
                    {log.data.level}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    {log.timestamp.toLocaleTimeString()}
                  </span>
                  <span className="text-sm flex-1">{log.data.message}</span>
                  {log.data.service && (
                    <Badge variant="outline" className="text-xs">
                      {log.data.service}
                    </Badge>
                  )}
                </div>
              ))}
              {startupService.logs.length === 0 && (
                <p className="text-center text-muted-foreground py-4">
                  No recent activity
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default StartupDashboardPage;
