/**
 * Service Controls Component
 * Control buttons and actions for individual services
 */

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { 
  Play, 
  Square, 
  RotateCcw, 
  Pause,
  Settings,
  MoreVertical,
  Eye,
  FileText,
  AlertTriangle
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { ServiceControlsProps } from './types';
import { ServiceStatus } from '@/types';

const ServiceControls: React.FC<ServiceControlsProps> = ({
  serviceName,
  status,
  onStart,
  onStop,
  onRestart,
  onPause,
  onConfigure,
  onViewLogs,
  onViewDetails,
  disabled = false,
  size = 'default',
  variant = 'default',
  className
}) => {
  const [isLoading, setIsLoading] = useState<string | null>(null);

  const handleAction = async (action: string, callback?: () => void | Promise<void>) => {
    if (!callback || disabled) return;
    
    setIsLoading(action);
    try {
      await callback();
    } catch (error) {
      console.error(`Failed to ${action} service ${serviceName}:`, error);
    } finally {
      setIsLoading(null);
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

  const canStart = status === ServiceStatus.STOPPED || status === ServiceStatus.ERROR;
  const canStop = status === ServiceStatus.RUNNING || status === ServiceStatus.STARTING;
  const canRestart = status === ServiceStatus.RUNNING;
  const canPause = status === ServiceStatus.RUNNING && onPause;

  if (variant === 'compact') {
    return (
      <div className={cn('flex items-center space-x-2', className)}>
        {/* Status Indicator */}
        <div className={cn(
          'w-2 h-2 rounded-full',
          getStatusColor(status)
        )} />
        
        {/* Primary Action Button */}
        {canStart && (
          <Button
            size={size}
            onClick={() => handleAction('start', onStart)}
            disabled={disabled || isLoading === 'start'}
          >
            <Play className="h-3 w-3" />
          </Button>
        )}
        
        {canStop && (
          <Button
            size={size}
            variant="outline"
            onClick={() => handleAction('stop', onStop)}
            disabled={disabled || isLoading === 'stop'}
          >
            <Square className="h-3 w-3" />
          </Button>
        )}

        {/* More Actions */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size={size} variant="ghost">
              <MoreVertical className="h-3 w-3" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>{serviceName}</DropdownMenuLabel>
            <DropdownMenuSeparator />
            
            {canRestart && (
              <DropdownMenuItem onClick={() => handleAction('restart', onRestart)}>
                <RotateCcw className="h-4 w-4 mr-2" />
                Restart
              </DropdownMenuItem>
            )}
            
            {canPause && (
              <DropdownMenuItem onClick={() => handleAction('pause', onPause)}>
                <Pause className="h-4 w-4 mr-2" />
                Pause
              </DropdownMenuItem>
            )}
            
            <DropdownMenuSeparator />
            
            {onViewDetails && (
              <DropdownMenuItem onClick={() => handleAction('view-details', onViewDetails)}>
                <Eye className="h-4 w-4 mr-2" />
                View Details
              </DropdownMenuItem>
            )}
            
            {onViewLogs && (
              <DropdownMenuItem onClick={() => handleAction('view-logs', onViewLogs)}>
                <FileText className="h-4 w-4 mr-2" />
                View Logs
              </DropdownMenuItem>
            )}
            
            {onConfigure && (
              <DropdownMenuItem onClick={() => handleAction('configure', onConfigure)}>
                <Settings className="h-4 w-4 mr-2" />
                Configure
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    );
  }

  if (variant === 'minimal') {
    return (
      <div className={cn('flex items-center space-x-1', className)}>
        {canStart && (
          <Button
            size="sm"
            onClick={() => handleAction('start', onStart)}
            disabled={disabled || isLoading === 'start'}
          >
            <Play className="h-3 w-3" />
          </Button>
        )}
        
        {canStop && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('stop', onStop)}
            disabled={disabled || isLoading === 'stop'}
          >
            <Square className="h-3 w-3" />
          </Button>
        )}
        
        {canRestart && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('restart', onRestart)}
            disabled={disabled || isLoading === 'restart'}
          >
            <RotateCcw className="h-3 w-3" />
          </Button>
        )}
      </div>
    );
  }

  // Default variant - full controls
  return (
    <div className={cn('space-y-3', className)}>
      {/* Service Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <h4 className="font-medium">{serviceName}</h4>
          <Badge variant="outline" className={cn('text-xs', getStatusColor(status))}>
            {status}
          </Badge>
        </div>
        
        {status === ServiceStatus.ERROR && (
          <AlertTriangle className="h-4 w-4 text-red-500" />
        )}
      </div>

      {/* Primary Controls */}
      <div className="flex space-x-2">
        <Button
          onClick={() => handleAction('start', onStart)}
          disabled={!canStart || disabled || isLoading === 'start'}
          className="flex-1"
        >
          <Play className="h-4 w-4 mr-2" />
          {isLoading === 'start' ? 'Starting...' : 'Start'}
        </Button>
        
        <Button
          variant="outline"
          onClick={() => handleAction('stop', onStop)}
          disabled={!canStop || disabled || isLoading === 'stop'}
          className="flex-1"
        >
          <Square className="h-4 w-4 mr-2" />
          {isLoading === 'stop' ? 'Stopping...' : 'Stop'}
        </Button>
        
        <Button
          variant="outline"
          onClick={() => handleAction('restart', onRestart)}
          disabled={!canRestart || disabled || isLoading === 'restart'}
          className="flex-1"
        >
          <RotateCcw className="h-4 w-4 mr-2" />
          {isLoading === 'restart' ? 'Restarting...' : 'Restart'}
        </Button>
      </div>

      {/* Secondary Controls */}
      <div className="flex space-x-2">
        {canPause && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('pause', onPause)}
            disabled={disabled || isLoading === 'pause'}
          >
            <Pause className="h-3 w-3 mr-1" />
            {isLoading === 'pause' ? 'Pausing...' : 'Pause'}
          </Button>
        )}
        
        {onViewDetails && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('view-details', onViewDetails)}
            disabled={disabled}
          >
            <Eye className="h-3 w-3 mr-1" />
            Details
          </Button>
        )}
        
        {onViewLogs && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('view-logs', onViewLogs)}
            disabled={disabled}
          >
            <FileText className="h-3 w-3 mr-1" />
            Logs
          </Button>
        )}
        
        {onConfigure && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('configure', onConfigure)}
            disabled={disabled}
          >
            <Settings className="h-3 w-3 mr-1" />
            Configure
          </Button>
        )}
      </div>
    </div>
  );
};

export default ServiceControls;
