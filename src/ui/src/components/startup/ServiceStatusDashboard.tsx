/**
 * Service Status Dashboard Component
 * Real-time service status visualization with control actions
 */

import React, { useState, useEffect } from 'react';
import { useServices, useServiceAction } from '@/hooks/useStartupService';
import { websocketService } from '@/services/websocket';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Play,
  Square,
  RotateCcw,
  Settings,
  RefreshCw,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Pause,
  MoreVertical,
  Eye,
  FileText,
  TrendingUp,
  Zap,
  Database,
  Server
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { ServiceStatusDashboardProps } from './types';
import { ServiceStatus, SystemHealthStatus } from '@/types';
import { HealthIndicator } from './HealthIndicator';

const ServiceStatusDashboard: React.FC<ServiceStatusDashboardProps> = ({
  services: propServices,
  systemStatus: propSystemStatus,
  onServiceAction,
  onRefresh,
  className
}) => {
  const [refreshing, setRefreshing] = useState(false);
  const [selectedService, setSelectedService] = useState<string | null>(null);
  const [serviceDetailsOpen, setServiceDetailsOpen] = useState(false);
  const [detailsService, setDetailsService] = useState<ServiceStatusInfo | null>(null);

  // Use React Query hooks for data fetching
  const { data: fetchedServices, isLoading, refetch } = useServices();
  const serviceAction = useServiceAction();

  // Use fetched data if available, otherwise fall back to props
  const services = fetchedServices || propServices;
  const systemStatus = propSystemStatus;

  // Set up WebSocket event listeners for real-time updates
  useEffect(() => {
    const handleServiceStatusUpdate = (data: any) => {
      console.log('Service status update received:', data);
      // The React Query cache will be updated automatically
      refetch();
    };

    const handleStartupProgress = (data: any) => {
      console.log('Startup progress update:', data);
      refetch();
    };

    // Subscribe to WebSocket events
    websocketService.on('startup_service_status_update', handleServiceStatusUpdate);
    websocketService.on('startup_progress_update', handleStartupProgress);

    // Subscribe to startup events when component mounts
    websocketService.subscribeToStartupEvents();

    return () => {
      websocketService.off('startup_service_status_update', handleServiceStatusUpdate);
      websocketService.off('startup_progress_update', handleStartupProgress);
    };
  }, [refetch]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await refetch();
      if (onRefresh) {
        await onRefresh();
      }
    } catch (error) {
      console.error('Failed to refresh services:', error);
    } finally {
      setTimeout(() => setRefreshing(false), 1000);
    }
  };

  const handleServiceActionClick = async (serviceName: string, action: 'start' | 'stop' | 'restart') => {
    try {
      await serviceAction.mutateAsync({
        serviceName,
        action: { action }
      });

      // Also call the prop callback if provided
      if (onServiceAction) {
        onServiceAction(serviceName, action);
      }
    } catch (error) {
      console.error(`Failed to ${action} service ${serviceName}:`, error);
    }
  };

  const getStatusIcon = (status: ServiceStatus) => {
    switch (status) {
      case ServiceStatus.RUNNING:
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case ServiceStatus.STARTING:
        return <Clock className="h-4 w-4 text-yellow-500 animate-spin" />;
      case ServiceStatus.STOPPING:
        return <Clock className="h-4 w-4 text-orange-500" />;
      case ServiceStatus.STOPPED:
        return <Square className="h-4 w-4 text-gray-500" />;
      case ServiceStatus.ERROR:
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: ServiceStatus) => {
    switch (status) {
      case ServiceStatus.RUNNING:
        return 'bg-green-500';
      case ServiceStatus.STARTING:
        return 'bg-yellow-500';
      case ServiceStatus.STOPPING:
        return 'bg-orange-500';
      case ServiceStatus.STOPPED:
        return 'bg-gray-500';
      case ServiceStatus.ERROR:
        return 'bg-red-500';
      default:
        return 'bg-gray-400';
    }
  };

  const getHealthScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatUptime = (seconds?: number) => {
    if (!seconds) return 'N/A';

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const getServiceTypeIcon = (serviceName: string) => {
    const name = serviceName.toLowerCase();
    if (name.includes('postgres') || name.includes('database')) {
      return <Database className="h-4 w-4 text-blue-500" />;
    }
    if (name.includes('redis') || name.includes('cache')) {
      return <Server className="h-4 w-4 text-red-500" />;
    }
    if (name.includes('ollama') || name.includes('ai')) {
      return <Zap className="h-4 w-4 text-purple-500" />;
    }
    return <Activity className="h-4 w-4 text-gray-500" />;
  };

  const openServiceDetails = (service: ServiceStatusInfo) => {
    setDetailsService(service);
    setServiceDetailsOpen(true);
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* System Overview */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <div>
            <CardTitle className="text-2xl font-bold">System Status</CardTitle>
            <CardDescription>
              Overall system health and service status overview
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
            className="ml-auto"
          >
            <RefreshCw className={cn("h-4 w-4 mr-2", refreshing && "animate-spin")} />
            Refresh
          </Button>
        </CardHeader>
        <CardContent>
          {systemStatus ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center space-x-3">
                <HealthIndicator 
                  status={systemStatus.overall_status as any}
                  healthScore={systemStatus.health_score}
                  size="lg"
                  showLabel
                />
                <div>
                  <p className="text-sm font-medium">Overall Health</p>
                  <p className={cn("text-2xl font-bold", getHealthScoreColor(systemStatus.health_score))}>
                    {Math.round(systemStatus.health_score * 100)}%
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <Activity className="h-8 w-8 text-blue-500" />
                <div>
                  <p className="text-sm font-medium">Active Services</p>
                  <p className="text-2xl font-bold">
                    {services.filter(s => s.status === ServiceStatus.RUNNING).length}/{services.length}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <Clock className="h-8 w-8 text-purple-500" />
                <div>
                  <p className="text-sm font-medium">Active Sequences</p>
                  <p className="text-2xl font-bold">{systemStatus.active_sequences.length}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>System status not available</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Loading State */}
      {isLoading && (
        <Card>
          <CardContent className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading services...</p>
          </CardContent>
        </Card>
      )}

      {/* Services Grid */}
      {!isLoading && services && services.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {services.map((service) => (
          <Card 
            key={service.service_name}
            className={cn(
              "transition-all duration-200 hover:shadow-md cursor-pointer",
              selectedService === service.service_name && "ring-2 ring-blue-500"
            )}
            onClick={() => setSelectedService(
              selectedService === service.service_name ? null : service.service_name
            )}
          >
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {getServiceTypeIcon(service.service_name)}
                  {getStatusIcon(service.status)}
                  <CardTitle className="text-lg">{service.service_name}</CardTitle>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant="outline" className={cn("text-xs", getStatusColor(service.status))}>
                    {service.status}
                  </Badge>

                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuLabel>Service Actions</DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem onClick={() => openServiceDetails(service)}>
                        <Eye className="h-4 w-4 mr-2" />
                        View Details
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <FileText className="h-4 w-4 mr-2" />
                        View Logs
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <TrendingUp className="h-4 w-4 mr-2" />
                        View Metrics
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem>
                        <Settings className="h-4 w-4 mr-2" />
                        Configure
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
              <div className="flex items-center space-x-4 text-sm text-gray-600">
                <span>Health: {Math.round(service.health_score * 100)}%</span>
                <span>Uptime: {formatUptime(service.uptime_seconds)}</span>
              </div>
            </CardHeader>
            
            <CardContent className="pt-0">
              {/* Health Score Progress */}
              <div className="mb-4">
                <div className="flex justify-between text-xs mb-1">
                  <span>Health Score</span>
                  <span className={getHealthScoreColor(service.health_score)}>
                    {Math.round(service.health_score * 100)}%
                  </span>
                </div>
                <Progress 
                  value={service.health_score * 100} 
                  className="h-2"
                />
              </div>

              {/* Error Message */}
              {service.error_message && (
                <div className="mb-4 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                  {service.error_message}
                </div>
              )}

              {/* Service Controls */}
              <div className="flex space-x-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleServiceActionClick(service.service_name, 'start');
                  }}
                  disabled={service.status === ServiceStatus.RUNNING || service.status === ServiceStatus.STARTING}
                  className="flex-1"
                >
                  <Play className="h-3 w-3 mr-1" />
                  Start
                </Button>
                
                <Button
                  size="sm"
                  variant="outline"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleServiceActionClick(service.service_name, 'stop');
                  }}
                  disabled={service.status === ServiceStatus.STOPPED || service.status === ServiceStatus.STOPPING}
                  className="flex-1"
                >
                  <Square className="h-3 w-3 mr-1" />
                  Stop
                </Button>
                
                <Button
                  size="sm"
                  variant="outline"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleServiceActionClick(service.service_name, 'restart');
                  }}
                  disabled={service.status === ServiceStatus.STARTING || service.status === ServiceStatus.STOPPING}
                >
                  <RotateCcw className="h-3 w-3" />
                </Button>
              </div>

              {/* Expanded Details */}
              {selectedService === service.service_name && (
                <>
                  <Separator className="my-4" />
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Last Check:</span>
                      <span>{new Date(service.last_check).toLocaleTimeString()}</span>
                    </div>
                    {Object.entries(service.metrics).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-gray-600 capitalize">{key.replace('_', ' ')}:</span>
                        <span>{typeof value === 'number' ? value.toFixed(2) : String(value)}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </CardContent>
          </Card>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && (!services || services.length === 0) && (
        <Card>
          <CardContent className="text-center py-12">
            <Activity className="h-16 w-16 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-semibold mb-2">No Services Found</h3>
            <p className="text-gray-600 mb-4">
              No services are currently configured or running.
            </p>
            <Button onClick={handleRefresh}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh Services
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Service Details Modal */}
      <Dialog open={serviceDetailsOpen} onOpenChange={setServiceDetailsOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              {detailsService && getServiceTypeIcon(detailsService.service_name)}
              <span>{detailsService?.service_name} Details</span>
            </DialogTitle>
            <DialogDescription>
              Detailed information and metrics for this service
            </DialogDescription>
          </DialogHeader>

          {detailsService && (
            <div className="space-y-6">
              {/* Service Status Overview */}
              <div className="grid grid-cols-2 gap-4">
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(detailsService.status)}
                      <div>
                        <p className="text-sm font-medium">Status</p>
                        <p className="text-lg font-bold">{detailsService.status}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <TrendingUp className="h-5 w-5 text-green-500" />
                      <div>
                        <p className="text-sm font-medium">Health Score</p>
                        <p className={cn("text-lg font-bold", getHealthScoreColor(detailsService.health_score))}>
                          {Math.round(detailsService.health_score * 100)}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Service Metrics */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Service Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Uptime</p>
                      <p className="text-lg font-semibold">{formatUptime(detailsService.uptime_seconds)}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Last Check</p>
                      <p className="text-lg font-semibold">
                        {new Date(detailsService.last_check).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>

                  {Object.keys(detailsService.metrics).length > 0 && (
                    <div className="mt-4">
                      <p className="text-sm font-medium text-muted-foreground mb-2">Performance Metrics</p>
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(detailsService.metrics).map(([key, value]) => (
                          <div key={key} className="flex justify-between p-2 bg-gray-50 rounded">
                            <span className="text-sm capitalize">{key.replace('_', ' ')}</span>
                            <span className="text-sm font-medium">
                              {typeof value === 'number' ? value.toFixed(2) : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Error Information */}
              {detailsService.error_message && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg text-red-600">Error Information</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="p-3 bg-red-50 border border-red-200 rounded">
                      <p className="text-sm text-red-700">{detailsService.error_message}</p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Service Actions */}
              <div className="flex space-x-2">
                <Button
                  onClick={() => handleServiceActionClick(detailsService.service_name, 'start')}
                  disabled={detailsService.status === ServiceStatus.RUNNING}
                  className="flex-1"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start Service
                </Button>

                <Button
                  variant="outline"
                  onClick={() => handleServiceActionClick(detailsService.service_name, 'stop')}
                  disabled={detailsService.status === ServiceStatus.STOPPED}
                  className="flex-1"
                >
                  <Square className="h-4 w-4 mr-2" />
                  Stop Service
                </Button>

                <Button
                  variant="outline"
                  onClick={() => handleServiceActionClick(detailsService.service_name, 'restart')}
                  disabled={detailsService.status === ServiceStatus.STARTING || detailsService.status === ServiceStatus.STOPPING}
                  className="flex-1"
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Restart
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ServiceStatusDashboard;
