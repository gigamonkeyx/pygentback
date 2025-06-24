/**
 * Sequence Controls Component
 * Control buttons and actions for startup sequences
 */

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { 
  Play, 
  Square, 
  Pause,
  RotateCcw,
  Settings,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { SequenceControlsProps } from './types';
import { SequenceStatus } from '@/types';

const SequenceControls: React.FC<SequenceControlsProps> = ({
  sequenceName,
  status,
  progress = 0,
  services = [],
  completedServices = [],
  onStart,
  onStop,
  onPause,
  onResume,
  onRestart,
  onConfigure,
  executionMode = 'sequential',
  onExecutionModeChange,
  timeoutSeconds = 300,
  onTimeoutChange,
  disabled = false,
  className
}) => {
  const [isLoading, setIsLoading] = useState<string | null>(null);

  const handleAction = async (action: string, callback?: () => void | Promise<void>) => {
    if (!callback || disabled) return;
    
    setIsLoading(action);
    try {
      await callback();
    } catch (error) {
      console.error(`Failed to ${action} sequence ${sequenceName}:`, error);
    } finally {
      setIsLoading(null);
    }
  };

  const getStatusIcon = (status: SequenceStatus) => {
    switch (status) {
      case SequenceStatus.RUNNING:
        return <Play className="h-4 w-4 text-blue-500 animate-pulse" />;
      case SequenceStatus.COMPLETED:
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case SequenceStatus.FAILED:
        return <XCircle className="h-4 w-4 text-red-500" />;
      case SequenceStatus.CANCELLED:
        return <XCircle className="h-4 w-4 text-orange-500" />;
      case SequenceStatus.PENDING:
        return <Clock className="h-4 w-4 text-gray-500" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: SequenceStatus) => {
    switch (status) {
      case SequenceStatus.RUNNING:
        return 'bg-blue-500';
      case SequenceStatus.COMPLETED:
        return 'bg-green-500';
      case SequenceStatus.FAILED:
        return 'bg-red-500';
      case SequenceStatus.CANCELLED:
        return 'bg-orange-500';
      case SequenceStatus.PENDING:
        return 'bg-gray-500';
      default:
        return 'bg-gray-400';
    }
  };

  const canStart = status === SequenceStatus.PENDING || status === SequenceStatus.FAILED || status === SequenceStatus.CANCELLED;
  const canStop = status === SequenceStatus.RUNNING;
  const canPause = status === SequenceStatus.RUNNING && onPause;
  const canResume = status === SequenceStatus.PAUSED && onResume;
  const canRestart = (status === SequenceStatus.COMPLETED || status === SequenceStatus.FAILED) && onRestart;

  const formatTimeout = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  return (
    <div className={cn('space-y-4', className)}>
      {/* Sequence Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getStatusIcon(status)}
          <div>
            <h3 className="font-semibold">{sequenceName}</h3>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className={cn('text-xs', getStatusColor(status))}>
                {status}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {services.length} services
              </span>
            </div>
          </div>
        </div>
        
        {onConfigure && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('configure', onConfigure)}
            disabled={disabled}
          >
            <Settings className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Progress */}
      {status === SequenceStatus.RUNNING && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <Progress value={progress} className="h-2" />
          <div className="text-xs text-muted-foreground">
            {completedServices.length} of {services.length} services completed
          </div>
        </div>
      )}

      {/* Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-sm font-medium">Execution Mode</label>
          <Select
            value={executionMode}
            onValueChange={onExecutionModeChange}
            disabled={disabled || status === SequenceStatus.RUNNING}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sequential">Sequential</SelectItem>
              <SelectItem value="parallel">Parallel</SelectItem>
            </SelectContent>
          </Select>
        </div>
        
        <div>
          <label className="text-sm font-medium">Timeout</label>
          <Select
            value={timeoutSeconds.toString()}
            onValueChange={(value) => onTimeoutChange?.(parseInt(value))}
            disabled={disabled || status === SequenceStatus.RUNNING}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="60">1 minute</SelectItem>
              <SelectItem value="300">5 minutes</SelectItem>
              <SelectItem value="600">10 minutes</SelectItem>
              <SelectItem value="1800">30 minutes</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Primary Controls */}
      <div className="flex space-x-2">
        {canStart && (
          <Button
            onClick={() => handleAction('start', onStart)}
            disabled={disabled || isLoading === 'start'}
            className="flex-1"
          >
            <Play className="h-4 w-4 mr-2" />
            {isLoading === 'start' ? 'Starting...' : 'Start Sequence'}
          </Button>
        )}
        
        {canResume && (
          <Button
            onClick={() => handleAction('resume', onResume)}
            disabled={disabled || isLoading === 'resume'}
            className="flex-1"
          >
            <Play className="h-4 w-4 mr-2" />
            {isLoading === 'resume' ? 'Resuming...' : 'Resume'}
          </Button>
        )}
        
        {canStop && (
          <Button
            variant="outline"
            onClick={() => handleAction('stop', onStop)}
            disabled={disabled || isLoading === 'stop'}
            className="flex-1"
          >
            <Square className="h-4 w-4 mr-2" />
            {isLoading === 'stop' ? 'Stopping...' : 'Stop'}
          </Button>
        )}
        
        {canPause && (
          <Button
            variant="outline"
            onClick={() => handleAction('pause', onPause)}
            disabled={disabled || isLoading === 'pause'}
          >
            <Pause className="h-4 w-4 mr-2" />
            {isLoading === 'pause' ? 'Pausing...' : 'Pause'}
          </Button>
        )}
        
        {canRestart && (
          <Button
            variant="outline"
            onClick={() => handleAction('restart', onRestart)}
            disabled={disabled || isLoading === 'restart'}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            {isLoading === 'restart' ? 'Restarting...' : 'Restart'}
          </Button>
        )}
      </div>

      {/* Service List */}
      {services.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Services in Sequence</h4>
          <div className="flex flex-wrap gap-2">
            {services.map((service, index) => (
              <Badge
                key={service}
                variant={completedServices.includes(service) ? "default" : "secondary"}
                className="text-xs"
              >
                {index + 1}. {service}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Status Information */}
      <div className="text-xs text-muted-foreground space-y-1">
        <div>Mode: {executionMode}</div>
        <div>Timeout: {formatTimeout(timeoutSeconds)}</div>
        {status === SequenceStatus.RUNNING && (
          <div>Estimated completion: {Math.round((100 - progress) / 10)} minutes</div>
        )}
      </div>
    </div>
  );
};

export default SequenceControls;
