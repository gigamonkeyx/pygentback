/**
 * Progress Tracker Component
 * Visual progress tracking for startup sequences and service operations
 */

import React from 'react';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  CheckCircle, 
  Clock, 
  XCircle, 
  AlertTriangle,
  Play,
  Pause
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { ProgressTrackerProps } from './types';

const ProgressTracker: React.FC<ProgressTrackerProps> = ({
  steps,
  currentStep,
  progress,
  status = 'running',
  showStepDetails = true,
  orientation = 'horizontal',
  className
}) => {
  const getStepStatus = (stepIndex: number) => {
    if (stepIndex < currentStep) return 'completed';
    if (stepIndex === currentStep) return status;
    return 'pending';
  };

  const getStepIcon = (stepStatus: string) => {
    switch (stepStatus) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'running':
        return <Clock className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'paused':
        return <Pause className="h-4 w-4 text-yellow-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStepColor = (stepStatus: string) => {
    switch (stepStatus) {
      case 'completed':
        return 'border-green-500 bg-green-50';
      case 'running':
        return 'border-blue-500 bg-blue-50';
      case 'failed':
        return 'border-red-500 bg-red-50';
      case 'paused':
        return 'border-yellow-500 bg-yellow-50';
      default:
        return 'border-gray-300 bg-gray-50';
    }
  };

  const getStatusBadgeVariant = (stepStatus: string) => {
    switch (stepStatus) {
      case 'completed':
        return 'default';
      case 'running':
        return 'default';
      case 'failed':
        return 'destructive';
      case 'paused':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  const overallProgress = steps.length > 0 ? ((currentStep / steps.length) * 100) : 0;
  const displayProgress = progress !== undefined ? progress : overallProgress;

  if (orientation === 'vertical') {
    return (
      <div className={cn('space-y-4', className)}>
        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Overall Progress</span>
            <span className="text-sm text-muted-foreground">
              {Math.round(displayProgress)}%
            </span>
          </div>
          <Progress value={displayProgress} className="h-2" />
        </div>

        {/* Step List */}
        {showStepDetails && (
          <div className="space-y-3">
            {steps.map((step, index) => {
              const stepStatus = getStepStatus(index);
              return (
                <div
                  key={index}
                  className={cn(
                    'flex items-start space-x-3 p-3 rounded-lg border-l-4',
                    getStepColor(stepStatus)
                  )}
                >
                  <div className="flex-shrink-0 mt-0.5">
                    {getStepIcon(stepStatus)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-medium">{step.name}</h4>
                      <Badge variant={getStatusBadgeVariant(stepStatus)} className="text-xs">
                        {stepStatus}
                      </Badge>
                    </div>
                    
                    {step.description && (
                      <p className="text-xs text-muted-foreground mt-1">
                        {step.description}
                      </p>
                    )}
                    
                    {step.duration && stepStatus === 'completed' && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Completed in {step.duration}ms
                      </p>
                    )}
                    
                    {step.error && stepStatus === 'failed' && (
                      <p className="text-xs text-red-600 mt-1">
                        Error: {step.error}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  // Horizontal orientation
  return (
    <div className={cn('space-y-4', className)}>
      {/* Overall Progress */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Progress</span>
          <span className="text-sm text-muted-foreground">
            {currentStep + 1} of {steps.length} steps ({Math.round(displayProgress)}%)
          </span>
        </div>
        <Progress value={displayProgress} className="h-2" />
      </div>

      {/* Step Timeline */}
      {showStepDetails && (
        <div className="relative">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => {
              const stepStatus = getStepStatus(index);
              const isLast = index === steps.length - 1;
              
              return (
                <div key={index} className="flex items-center">
                  {/* Step Circle */}
                  <div className={cn(
                    'flex items-center justify-center w-8 h-8 rounded-full border-2',
                    stepStatus === 'completed' && 'border-green-500 bg-green-500',
                    stepStatus === 'running' && 'border-blue-500 bg-blue-500',
                    stepStatus === 'failed' && 'border-red-500 bg-red-500',
                    stepStatus === 'paused' && 'border-yellow-500 bg-yellow-500',
                    stepStatus === 'pending' && 'border-gray-300 bg-white'
                  )}>
                    {stepStatus === 'completed' && (
                      <CheckCircle className="h-4 w-4 text-white" />
                    )}
                    {stepStatus === 'running' && (
                      <Clock className="h-4 w-4 text-white animate-spin" />
                    )}
                    {stepStatus === 'failed' && (
                      <XCircle className="h-4 w-4 text-white" />
                    )}
                    {stepStatus === 'paused' && (
                      <Pause className="h-4 w-4 text-white" />
                    )}
                    {stepStatus === 'pending' && (
                      <span className="text-xs font-medium text-gray-500">
                        {index + 1}
                      </span>
                    )}
                  </div>

                  {/* Connector Line */}
                  {!isLast && (
                    <div className={cn(
                      'flex-1 h-0.5 mx-2',
                      index < currentStep ? 'bg-green-500' : 'bg-gray-300'
                    )} />
                  )}
                </div>
              );
            })}
          </div>

          {/* Step Labels */}
          <div className="flex items-start justify-between mt-2">
            {steps.map((step, index) => {
              const stepStatus = getStepStatus(index);
              
              return (
                <div key={index} className="flex flex-col items-center max-w-24">
                  <span className={cn(
                    'text-xs font-medium text-center',
                    stepStatus === 'completed' && 'text-green-600',
                    stepStatus === 'running' && 'text-blue-600',
                    stepStatus === 'failed' && 'text-red-600',
                    stepStatus === 'paused' && 'text-yellow-600',
                    stepStatus === 'pending' && 'text-gray-500'
                  )}>
                    {step.name}
                  </span>
                  
                  {step.duration && stepStatus === 'completed' && (
                    <span className="text-xs text-muted-foreground">
                      {step.duration}ms
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProgressTracker;
