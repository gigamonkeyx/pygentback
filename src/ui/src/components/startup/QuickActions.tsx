/**
 * Quick Actions Component
 * Common quick action buttons for startup service management
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
  Zap, 
  Square, 
  RotateCcw,
  Settings,
  FileText,
  BarChart3,
  TestTube,
  Download,
  Upload,
  RefreshCw,
  HelpCircle,
  Plus,
  ChevronDown
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { QuickActionsProps } from './types';

const QuickActions: React.FC<QuickActionsProps> = ({
  onStartAll,
  onStopAll,
  onRestartAll,
  onRefreshStatus,
  onOpenConfiguration,
  onOpenMonitoring,
  onRunTests,
  onExportLogs,
  onImportProfile,
  onCreateProfile,
  onOpenHelp,
  activeServices = 0,
  totalServices = 0,
  isLoading = false,
  variant = 'default',
  className
}) => {
  const [loadingAction, setLoadingAction] = useState<string | null>(null);

  const handleAction = async (actionName: string, callback?: () => void | Promise<void>) => {
    if (!callback || isLoading) return;
    
    setLoadingAction(actionName);
    try {
      await callback();
    } catch (error) {
      console.error(`Failed to execute ${actionName}:`, error);
    } finally {
      setLoadingAction(null);
    }
  };

  const isActionLoading = (action: string) => loadingAction === action || isLoading;

  if (variant === 'compact') {
    return (
      <div className={cn('flex items-center space-x-2', className)}>
        {/* Service Status */}
        <Badge variant="outline" className="text-xs">
          {activeServices}/{totalServices} active
        </Badge>

        {/* Primary Actions */}
        <Button
          size="sm"
          onClick={() => handleAction('start-all', onStartAll)}
          disabled={isActionLoading('start-all')}
        >
          <Zap className="h-3 w-3 mr-1" />
          Start All
        </Button>

        <Button
          size="sm"
          variant="outline"
          onClick={() => handleAction('stop-all', onStopAll)}
          disabled={isActionLoading('stop-all')}
        >
          <Square className="h-3 w-3 mr-1" />
          Stop All
        </Button>

        {/* More Actions */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="sm" variant="ghost">
              <ChevronDown className="h-3 w-3" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Quick Actions</DropdownMenuLabel>
            <DropdownMenuSeparator />
            
            {onRestartAll && (
              <DropdownMenuItem onClick={() => handleAction('restart-all', onRestartAll)}>
                <RotateCcw className="h-4 w-4 mr-2" />
                Restart All
              </DropdownMenuItem>
            )}
            
            {onRefreshStatus && (
              <DropdownMenuItem onClick={() => handleAction('refresh', onRefreshStatus)}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh Status
              </DropdownMenuItem>
            )}
            
            <DropdownMenuSeparator />
            
            {onOpenConfiguration && (
              <DropdownMenuItem onClick={() => handleAction('config', onOpenConfiguration)}>
                <Settings className="h-4 w-4 mr-2" />
                Configuration
              </DropdownMenuItem>
            )}
            
            {onOpenMonitoring && (
              <DropdownMenuItem onClick={() => handleAction('monitoring', onOpenMonitoring)}>
                <BarChart3 className="h-4 w-4 mr-2" />
                Monitoring
              </DropdownMenuItem>
            )}
            
            {onRunTests && (
              <DropdownMenuItem onClick={() => handleAction('tests', onRunTests)}>
                <TestTube className="h-4 w-4 mr-2" />
                Run Tests
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
        <Button
          size="sm"
          onClick={() => handleAction('start-all', onStartAll)}
          disabled={isActionLoading('start-all')}
        >
          <Zap className="h-3 w-3" />
        </Button>

        <Button
          size="sm"
          variant="outline"
          onClick={() => handleAction('stop-all', onStopAll)}
          disabled={isActionLoading('stop-all')}
        >
          <Square className="h-3 w-3" />
        </Button>

        {onRefreshStatus && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('refresh', onRefreshStatus)}
            disabled={isActionLoading('refresh')}
          >
            <RefreshCw className={cn('h-3 w-3', isActionLoading('refresh') && 'animate-spin')} />
          </Button>
        )}
      </div>
    );
  }

  // Default variant - full quick actions panel
  return (
    <div className={cn('space-y-4', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Quick Actions</h3>
        <Badge variant="outline" className="text-xs">
          {activeServices}/{totalServices} services active
        </Badge>
      </div>

      {/* Primary Service Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        <Button
          onClick={() => handleAction('start-all', onStartAll)}
          disabled={isActionLoading('start-all')}
          className="w-full"
        >
          <Zap className="h-4 w-4 mr-2" />
          {isActionLoading('start-all') ? 'Starting...' : 'Start All'}
        </Button>

        <Button
          variant="outline"
          onClick={() => handleAction('stop-all', onStopAll)}
          disabled={isActionLoading('stop-all')}
          className="w-full"
        >
          <Square className="h-4 w-4 mr-2" />
          {isActionLoading('stop-all') ? 'Stopping...' : 'Stop All'}
        </Button>

        {onRestartAll && (
          <Button
            variant="outline"
            onClick={() => handleAction('restart-all', onRestartAll)}
            disabled={isActionLoading('restart-all')}
            className="w-full"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            {isActionLoading('restart-all') ? 'Restarting...' : 'Restart All'}
          </Button>
        )}
      </div>

      {/* Navigation Actions */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {onOpenConfiguration && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('config', onOpenConfiguration)}
            disabled={isActionLoading('config')}
          >
            <Settings className="h-3 w-3 mr-1" />
            Config
          </Button>
        )}

        {onOpenMonitoring && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('monitoring', onOpenMonitoring)}
            disabled={isActionLoading('monitoring')}
          >
            <BarChart3 className="h-3 w-3 mr-1" />
            Monitor
          </Button>
        )}

        {onRunTests && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('tests', onRunTests)}
            disabled={isActionLoading('tests')}
          >
            <TestTube className="h-3 w-3 mr-1" />
            Tests
          </Button>
        )}

        {onRefreshStatus && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('refresh', onRefreshStatus)}
            disabled={isActionLoading('refresh')}
          >
            <RefreshCw className={cn('h-3 w-3 mr-1', isActionLoading('refresh') && 'animate-spin')} />
            Refresh
          </Button>
        )}
      </div>

      {/* Utility Actions */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {onExportLogs && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('export', onExportLogs)}
            disabled={isActionLoading('export')}
          >
            <Download className="h-3 w-3 mr-1" />
            Export
          </Button>
        )}

        {onImportProfile && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('import', onImportProfile)}
            disabled={isActionLoading('import')}
          >
            <Upload className="h-3 w-3 mr-1" />
            Import
          </Button>
        )}

        {onCreateProfile && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('create', onCreateProfile)}
            disabled={isActionLoading('create')}
          >
            <Plus className="h-3 w-3 mr-1" />
            Profile
          </Button>
        )}

        {onOpenHelp && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleAction('help', onOpenHelp)}
            disabled={isActionLoading('help')}
          >
            <HelpCircle className="h-3 w-3 mr-1" />
            Help
          </Button>
        )}
      </div>

      {/* Status Information */}
      <div className="text-xs text-muted-foreground">
        <div>Last updated: {new Date().toLocaleTimeString()}</div>
        {isLoading && (
          <div className="flex items-center space-x-1 mt-1">
            <RefreshCw className="h-3 w-3 animate-spin" />
            <span>Updating...</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuickActions;
