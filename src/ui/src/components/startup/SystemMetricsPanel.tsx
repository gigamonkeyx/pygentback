/**
 * System Metrics Panel Component
 * Real-time system metrics visualization and monitoring
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { 
  BarChart3, 
  Activity,
  Cpu,
  HardDrive,
  MemoryStick,
  Network,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Maximize2,
  Settings
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { SystemMetricsPanelProps, MetricConfig, ChartData } from './types';

const SystemMetricsPanel: React.FC<SystemMetricsPanelProps> = ({
  metrics,
  realtimeData,
  onMetricSelect,
  refreshInterval = 5000,
  className
}) => {
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [metricHistory, setMetricHistory] = useState<Record<string, ChartData[]>>({});

  // Simulate metric history for demonstration
  useEffect(() => {
    if (isAutoRefresh && metrics) {
      const interval = setInterval(() => {
        const timestamp = new Date();
        const newHistory = { ...metricHistory };
        
        Object.entries(metrics).forEach(([key, value]) => {
          if (typeof value === 'number') {
            if (!newHistory[key]) {
              newHistory[key] = [];
            }
            newHistory[key].push({
              timestamp,
              value,
              label: key
            });
            
            // Keep only last 100 data points
            if (newHistory[key].length > 100) {
              newHistory[key] = newHistory[key].slice(-100);
            }
          }
        });
        
        setMetricHistory(newHistory);
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [metrics, metricHistory, refreshInterval, isAutoRefresh]);

  const metricConfigs: MetricConfig[] = [
    {
      key: 'cpu_usage',
      label: 'CPU Usage',
      unit: '%',
      format: 'percentage',
      color: 'text-blue-600',
      threshold: { warning: 70, critical: 90 }
    },
    {
      key: 'memory_usage',
      label: 'Memory Usage',
      unit: '%',
      format: 'percentage',
      color: 'text-green-600',
      threshold: { warning: 80, critical: 95 }
    },
    {
      key: 'disk_usage',
      label: 'Disk Usage',
      unit: '%',
      format: 'percentage',
      color: 'text-orange-600',
      threshold: { warning: 85, critical: 95 }
    },
    {
      key: 'network_io',
      label: 'Network I/O',
      unit: 'MB/s',
      format: 'bytes',
      color: 'text-purple-600'
    }
  ];

  const getMetricIcon = (key: string) => {
    switch (key) {
      case 'cpu_usage':
        return <Cpu className="h-5 w-5 text-blue-500" />;
      case 'memory_usage':
        return <MemoryStick className="h-5 w-5 text-green-500" />;
      case 'disk_usage':
        return <HardDrive className="h-5 w-5 text-orange-500" />;
      case 'network_io':
        return <Network className="h-5 w-5 text-purple-500" />;
      default:
        return <Activity className="h-5 w-5 text-gray-500" />;
    }
  };

  const formatMetricValue = (value: number, format?: string, unit?: string) => {
    let formattedValue: string;
    
    switch (format) {
      case 'percentage':
        formattedValue = `${value.toFixed(1)}%`;
        break;
      case 'bytes':
        if (value >= 1024 * 1024 * 1024) {
          formattedValue = `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
        } else if (value >= 1024 * 1024) {
          formattedValue = `${(value / (1024 * 1024)).toFixed(2)} MB`;
        } else if (value >= 1024) {
          formattedValue = `${(value / 1024).toFixed(2)} KB`;
        } else {
          formattedValue = `${value.toFixed(2)} B`;
        }
        break;
      case 'duration':
        const hours = Math.floor(value / 3600);
        const minutes = Math.floor((value % 3600) / 60);
        const seconds = Math.floor(value % 60);
        formattedValue = `${hours}h ${minutes}m ${seconds}s`;
        break;
      default:
        formattedValue = value.toFixed(2);
        if (unit) formattedValue += ` ${unit}`;
    }
    
    return formattedValue;
  };

  const getMetricStatus = (value: number, threshold?: { warning: number; critical: number }) => {
    if (!threshold) return 'normal';
    if (value >= threshold.critical) return 'critical';
    if (value >= threshold.warning) return 'warning';
    return 'normal';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'critical':
        return 'text-red-600 bg-red-100';
      case 'warning':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-green-600 bg-green-100';
    }
  };

  const getTrendIcon = (current: number, previous?: number) => {
    if (!previous) return <Minus className="h-4 w-4 text-gray-400" />;
    
    if (current > previous) {
      return <TrendingUp className="h-4 w-4 text-red-500" />;
    } else if (current < previous) {
      return <TrendingDown className="h-4 w-4 text-green-500" />;
    } else {
      return <Minus className="h-4 w-4 text-gray-400" />;
    }
  };

  const renderMiniChart = (data: ChartData[]) => {
    if (data.length < 2) return null;
    
    const maxValue = Math.max(...data.map(d => d.value));
    const minValue = Math.min(...data.map(d => d.value));
    const range = maxValue - minValue || 1;
    
    const points = data.map((point, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = 100 - ((point.value - minValue) / range) * 100;
      return `${x},${y}`;
    }).join(' ');
    
    return (
      <svg className="w-full h-12" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          points={points}
          className="text-blue-500"
        />
      </svg>
    );
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* Metrics Overview */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-blue-500" />
              <span>System Metrics</span>
            </CardTitle>
            <CardDescription>
              Real-time system performance monitoring
            </CardDescription>
          </div>
          
          <div className="flex space-x-2">
            <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="5m">5 minutes</SelectItem>
                <SelectItem value="1h">1 hour</SelectItem>
                <SelectItem value="6h">6 hours</SelectItem>
                <SelectItem value="24h">24 hours</SelectItem>
              </SelectContent>
            </Select>
            
            <Button
              size="sm"
              variant={isAutoRefresh ? "default" : "outline"}
              onClick={() => setIsAutoRefresh(!isAutoRefresh)}
            >
              <RefreshCw className={cn("h-4 w-4", isAutoRefresh && "animate-spin")} />
            </Button>
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {metricConfigs.map((config) => {
              const value = metrics[config.key] as number;
              const history = metricHistory[config.key] || [];
              const previousValue = history.length > 1 ? history[history.length - 2].value : undefined;
              const status = getMetricStatus(value, config.threshold);
              
              if (value === undefined) return null;
              
              return (
                <Card 
                  key={config.key}
                  className={cn(
                    "cursor-pointer transition-all hover:shadow-md",
                    selectedMetric === config.key && "ring-2 ring-blue-500"
                  )}
                  onClick={() => {
                    setSelectedMetric(selectedMetric === config.key ? null : config.key);
                    onMetricSelect(config.key);
                  }}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {getMetricIcon(config.key)}
                        <span className="text-sm font-medium">{config.label}</span>
                      </div>
                      {getTrendIcon(value, previousValue)}
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className={cn("text-2xl font-bold", config.color)}>
                          {formatMetricValue(value, config.format, config.unit)}
                        </span>
                        <Badge variant="outline" className={getStatusColor(status)}>
                          {status}
                        </Badge>
                      </div>
                      
                      {config.format === 'percentage' && (
                        <Progress value={value} className="h-2" />
                      )}
                      
                      {history.length > 1 && (
                        <div className="h-12 mt-2">
                          {renderMiniChart(history)}
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Metrics */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="resources">Resources</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>System Overview</CardTitle>
              <CardDescription>
                Current system status and key metrics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(metrics).map(([key, value]) => (
                  <div key={key} className="text-center p-3 bg-gray-50 rounded">
                    <p className="text-sm font-medium text-muted-foreground capitalize">
                      {key.replace('_', ' ')}
                    </p>
                    <p className="text-lg font-bold">
                      {typeof value === 'number' ? value.toFixed(2) : String(value)}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>
                Detailed performance analysis and trends
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Performance charts coming soon</p>
                <p className="text-sm">Detailed performance visualization will be available here</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="resources" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Resource Usage</CardTitle>
              <CardDescription>
                CPU, memory, and disk utilization details
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {['cpu_usage', 'memory_usage', 'disk_usage'].map((key) => {
                  const value = metrics[key] as number;
                  const config = metricConfigs.find(c => c.key === key);
                  
                  if (!config || value === undefined) return null;
                  
                  return (
                    <div key={key} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {getMetricIcon(key)}
                          <span className="font-medium">{config.label}</span>
                        </div>
                        <span className="text-lg font-bold">
                          {formatMetricValue(value, config.format)}
                        </span>
                      </div>
                      <Progress value={value} className="h-3" />
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="network" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Network Activity</CardTitle>
              <CardDescription>
                Network I/O and connection statistics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <Network className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Network metrics coming soon</p>
                <p className="text-sm">Network activity visualization will be available here</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SystemMetricsPanel;
