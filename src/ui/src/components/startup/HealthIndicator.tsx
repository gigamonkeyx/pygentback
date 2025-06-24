/**
 * Health Indicator Component
 * Visual indicator for service and system health status
 */

import React from 'react';
import { 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  Clock, 
  HelpCircle,
  Activity,
  Zap
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { HealthIndicatorProps } from './types';
import { ServiceStatus, SequenceStatus, SystemHealthStatus } from '@/types';

export const HealthIndicator: React.FC<HealthIndicatorProps> = ({
  status,
  healthScore,
  lastCheck,
  errorMessage,
  size = 'md',
  showLabel = false,
  className
}) => {
  const getStatusConfig = (status: ServiceStatus | SequenceStatus | SystemHealthStatus) => {
    // Service Status
    if (Object.values(ServiceStatus).includes(status as ServiceStatus)) {
      switch (status as ServiceStatus) {
        case ServiceStatus.RUNNING:
          return {
            icon: CheckCircle,
            color: 'text-green-500',
            bgColor: 'bg-green-100',
            label: 'Running',
            pulse: false
          };
        case ServiceStatus.STARTING:
          return {
            icon: Clock,
            color: 'text-yellow-500',
            bgColor: 'bg-yellow-100',
            label: 'Starting',
            pulse: true
          };
        case ServiceStatus.STOPPING:
          return {
            icon: Clock,
            color: 'text-orange-500',
            bgColor: 'bg-orange-100',
            label: 'Stopping',
            pulse: true
          };
        case ServiceStatus.STOPPED:
          return {
            icon: XCircle,
            color: 'text-gray-500',
            bgColor: 'bg-gray-100',
            label: 'Stopped',
            pulse: false
          };
        case ServiceStatus.ERROR:
          return {
            icon: XCircle,
            color: 'text-red-500',
            bgColor: 'bg-red-100',
            label: 'Error',
            pulse: false
          };
        default:
          return {
            icon: HelpCircle,
            color: 'text-gray-400',
            bgColor: 'bg-gray-100',
            label: 'Unknown',
            pulse: false
          };
      }
    }

    // Sequence Status
    if (Object.values(SequenceStatus).includes(status as SequenceStatus)) {
      switch (status as SequenceStatus) {
        case SequenceStatus.RUNNING:
          return {
            icon: Activity,
            color: 'text-blue-500',
            bgColor: 'bg-blue-100',
            label: 'Running',
            pulse: true
          };
        case SequenceStatus.COMPLETED:
          return {
            icon: CheckCircle,
            color: 'text-green-500',
            bgColor: 'bg-green-100',
            label: 'Completed',
            pulse: false
          };
        case SequenceStatus.FAILED:
          return {
            icon: XCircle,
            color: 'text-red-500',
            bgColor: 'bg-red-100',
            label: 'Failed',
            pulse: false
          };
        case SequenceStatus.CANCELLED:
          return {
            icon: XCircle,
            color: 'text-orange-500',
            bgColor: 'bg-orange-100',
            label: 'Cancelled',
            pulse: false
          };
        case SequenceStatus.PENDING:
          return {
            icon: Clock,
            color: 'text-gray-500',
            bgColor: 'bg-gray-100',
            label: 'Pending',
            pulse: false
          };
        default:
          return {
            icon: HelpCircle,
            color: 'text-gray-400',
            bgColor: 'bg-gray-100',
            label: 'Unknown',
            pulse: false
          };
      }
    }

    // System Health Status
    switch (status as SystemHealthStatus) {
      case SystemHealthStatus.HEALTHY:
        return {
          icon: CheckCircle,
          color: 'text-green-500',
          bgColor: 'bg-green-100',
          label: 'Healthy',
          pulse: false
        };
      case SystemHealthStatus.DEGRADED:
        return {
          icon: AlertTriangle,
          color: 'text-yellow-500',
          bgColor: 'bg-yellow-100',
          label: 'Degraded',
          pulse: false
        };
      case SystemHealthStatus.UNHEALTHY:
        return {
          icon: XCircle,
          color: 'text-red-500',
          bgColor: 'bg-red-100',
          label: 'Unhealthy',
          pulse: false
        };
      default:
        return {
          icon: HelpCircle,
          color: 'text-gray-400',
          bgColor: 'bg-gray-100',
          label: 'Unknown',
          pulse: false
        };
    }
  };

  const getSizeClasses = (size: 'sm' | 'md' | 'lg') => {
    switch (size) {
      case 'sm':
        return {
          container: 'w-6 h-6',
          icon: 'h-3 w-3',
          text: 'text-xs'
        };
      case 'lg':
        return {
          container: 'w-12 h-12',
          icon: 'h-6 w-6',
          text: 'text-sm'
        };
      default:
        return {
          container: 'w-8 h-8',
          icon: 'h-4 w-4',
          text: 'text-sm'
        };
    }
  };

  const getHealthScoreColor = (score?: number) => {
    if (!score) return 'text-gray-500';
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const config = getStatusConfig(status);
  const sizeClasses = getSizeClasses(size);
  const Icon = config.icon;

  const formatLastCheck = (date?: Date) => {
    if (!date) return null;
    
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  return (
    <div className={cn('flex items-center space-x-2', className)}>
      {/* Status Indicator */}
      <div className={cn(
        'rounded-full flex items-center justify-center',
        config.bgColor,
        sizeClasses.container,
        config.pulse && 'animate-pulse'
      )}>
        <Icon className={cn(config.color, sizeClasses.icon)} />
      </div>

      {/* Label and Details */}
      {showLabel && (
        <div className="flex flex-col">
          <div className="flex items-center space-x-2">
            <span className={cn('font-medium', sizeClasses.text)}>
              {config.label}
            </span>
            {healthScore !== undefined && (
              <span className={cn(
                'font-semibold',
                sizeClasses.text,
                getHealthScoreColor(healthScore)
              )}>
                {Math.round(healthScore * 100)}%
              </span>
            )}
          </div>
          
          {(lastCheck || errorMessage) && (
            <div className={cn('text-gray-500', sizeClasses.text)}>
              {errorMessage ? (
                <span className="text-red-600 truncate max-w-32" title={errorMessage}>
                  {errorMessage}
                </span>
              ) : lastCheck ? (
                <span>{formatLastCheck(lastCheck)}</span>
              ) : null}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default HealthIndicator;
