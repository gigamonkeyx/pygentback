/**
 * Startup Orchestration Page
 * Service orchestration and sequence management interface
 */

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Cog, 
  Play, 
  Square, 
  Pause,
  RotateCcw,
  Plus,
  Settings,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle
} from 'lucide-react';
import { useStartupService } from '@/stores/appStore';
import { useSequences, useSequenceAction, useProfiles } from '@/hooks/useStartupService';
import { SequenceStatus } from '@/types';
import { StartupOrchestrator } from '@/components/startup';

const StartupOrchestrationPage: React.FC = () => {
  const { startupService } = useStartupService();
  const { data: sequences, isLoading, refetch } = useSequences();
  const { data: profiles } = useProfiles();
  const sequenceAction = useSequenceAction();
  const [selectedSequence, setSelectedSequence] = useState<string | null>(null);

  const getStatusIcon = (status: SequenceStatus) => {
    switch (status) {
      case SequenceStatus.RUNNING:
        return <Play className="h-4 w-4 text-blue-500" />;
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

  const handleSequenceAction = async (sequenceId: string, action: string) => {
    try {
      await sequenceAction.mutateAsync({
        sequenceId,
        action: { action } as any
      });
    } catch (error) {
      console.error(`Failed to ${action} sequence:`, error);
    }
  };

  return (
    <div className="h-full overflow-auto">
      <div className="container mx-auto p-6 space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center">
              <Cog className="h-8 w-8 mr-3 text-blue-500" />
              Service Orchestration
            </h1>
            <p className="text-muted-foreground mt-2">
              Manage startup sequences and service orchestration workflows
            </p>
          </div>
          
          <div className="flex space-x-2">
            <Button variant="outline" onClick={() => refetch()}>
              <RotateCcw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              New Sequence
            </Button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <Play className="h-8 w-8 text-blue-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-muted-foreground">Active Sequences</p>
                  <p className="text-2xl font-bold">
                    {sequences?.filter(s => s.status === SequenceStatus.RUNNING).length || 0}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <CheckCircle className="h-8 w-8 text-green-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-muted-foreground">Completed Today</p>
                  <p className="text-2xl font-bold">
                    {sequences?.filter(s => s.status === SequenceStatus.COMPLETED).length || 0}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <XCircle className="h-8 w-8 text-red-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-muted-foreground">Failed</p>
                  <p className="text-2xl font-bold">
                    {sequences?.filter(s => s.status === SequenceStatus.FAILED).length || 0}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <Clock className="h-8 w-8 text-gray-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-muted-foreground">Pending</p>
                  <p className="text-2xl font-bold">
                    {sequences?.filter(s => s.status === SequenceStatus.PENDING).length || 0}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Startup Orchestrator Component */}
        <StartupOrchestrator
          profiles={profiles || []}
          activeSequences={sequences?.filter(s => s.status === SequenceStatus.RUNNING) || []}
          onStartSequence={(request) => {
            console.log('Start sequence requested:', request);
          }}
          onStopSequence={(sequenceId) => {
            handleSequenceAction(sequenceId, 'stop');
          }}
          onCreateProfile={(profile) => {
            console.log('Create profile requested:', profile);
          }}
        />

        {/* Sequences List */}
        <Card>
          <CardHeader>
            <CardTitle>Startup Sequences</CardTitle>
            <CardDescription>
              Manage and monitor service startup sequences
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                <p className="mt-2 text-muted-foreground">Loading sequences...</p>
              </div>
            ) : sequences && sequences.length > 0 ? (
              <div className="space-y-4">
                {sequences.map((sequence) => (
                  <div
                    key={sequence.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-all ${
                      selectedSequence === sequence.id ? 'border-primary bg-primary/5' : 'hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedSequence(
                      selectedSequence === sequence.id ? null : sequence.id
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(sequence.status)}
                        <div>
                          <h3 className="font-semibold">{sequence.sequence_name}</h3>
                          <p className="text-sm text-muted-foreground">
                            {sequence.services.length} services • {sequence.environment}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline" className={getStatusColor(sequence.status)}>
                          {sequence.status}
                        </Badge>
                        
                        <div className="flex space-x-1">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSequenceAction(sequence.id, 'start');
                            }}
                            disabled={sequence.status === SequenceStatus.RUNNING}
                          >
                            <Play className="h-3 w-3" />
                          </Button>
                          
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSequenceAction(sequence.id, 'stop');
                            }}
                            disabled={sequence.status !== SequenceStatus.RUNNING}
                          >
                            <Square className="h-3 w-3" />
                          </Button>
                          
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              // Handle configure action
                            }}
                          >
                            <Settings className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {selectedSequence === sequence.id && (
                      <div className="mt-4 pt-4 border-t space-y-3">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="font-medium">Services:</span>
                            <div className="mt-1 space-y-1">
                              {sequence.services.map((service, index) => (
                                <Badge key={index} variant="secondary" className="mr-1">
                                  {service}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          
                          <div>
                            <span className="font-medium">Configuration:</span>
                            <div className="mt-1 space-y-1">
                              <p className="text-muted-foreground">
                                Parallel: {sequence.parallel_execution ? 'Yes' : 'No'}
                              </p>
                              <p className="text-muted-foreground">
                                Timeout: {sequence.timeout_seconds}s
                              </p>
                              <p className="text-muted-foreground">
                                Retries: {sequence.retry_attempts}
                              </p>
                            </div>
                          </div>
                        </div>

                        {sequence.started_at && (
                          <div className="text-sm">
                            <span className="font-medium">Started:</span>
                            <span className="ml-2 text-muted-foreground">
                              {new Date(sequence.started_at).toLocaleString()}
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <Cog className="h-16 w-16 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-semibold mb-2">No Sequences Found</h3>
                <p className="text-gray-600 mb-4">
                  Create your first startup sequence to begin orchestrating services.
                </p>
                <Button>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Sequence
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default StartupOrchestrationPage;
