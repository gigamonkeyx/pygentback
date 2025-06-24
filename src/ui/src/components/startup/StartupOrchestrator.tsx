/**
 * Startup Orchestrator Component
 * Advanced service orchestration interface with dependency visualization
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
  Pause,
  Settings,
  Plus,
  Trash2,
  ArrowRight,
  GitBranch,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Zap,
  Users,
  Timer
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { StartupOrchestratorProps } from './types';
import { SequenceStatus, ServiceStatus } from '@/types';
import { useStartupWithProfile, useSequenceAction } from '@/hooks/useStartupService';
import DependencyGraph from './DependencyGraph';

const StartupOrchestrator: React.FC<StartupOrchestratorProps> = ({
  profiles,
  activeSequences,
  onStartSequence,
  onStopSequence,
  onCreateProfile,
  className
}) => {
  const [selectedProfile, setSelectedProfile] = useState<string>('');
  const [customServices, setCustomServices] = useState<string[]>([]);
  const [parallelExecution, setParallelExecution] = useState(false);
  const [timeoutSeconds, setTimeoutSeconds] = useState(300);
  const [showDependencyGraph, setShowDependencyGraph] = useState(false);
  
  const startupWithProfile = useStartupWithProfile();
  const sequenceAction = useSequenceAction();

  const getSequenceStatusIcon = (status: SequenceStatus) => {
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

  const getSequenceProgress = (sequence: any) => {
    if (sequence.status === SequenceStatus.COMPLETED) return 100;
    if (sequence.status === SequenceStatus.FAILED || sequence.status === SequenceStatus.CANCELLED) return 0;
    if (sequence.status === SequenceStatus.PENDING) return 0;
    
    // Calculate progress based on completed services
    const totalServices = sequence.services.length;
    const completedServices = sequence.services_completed?.length || 0;
    return totalServices > 0 ? (completedServices / totalServices) * 100 : 0;
  };

  const handleStartSequence = async () => {
    if (!selectedProfile) return;
    
    try {
      await startupWithProfile.mutateAsync({
        profileId: selectedProfile,
        options: {
          parallel_execution: parallelExecution,
          timeout_seconds: timeoutSeconds,
          environment: 'development'
        }
      });
    } catch (error) {
      console.error('Failed to start sequence:', error);
    }
  };

  const handleStopSequence = async (sequenceId: string) => {
    try {
      await sequenceAction.mutateAsync({
        sequenceId,
        action: { action: 'stop' }
      });
    } catch (error) {
      console.error('Failed to stop sequence:', error);
    }
  };

  const renderDependencyGraph = (profile: any) => {
    if (!profile || !profile.startup_sequence) return null;

    return (
      <div className="space-y-4">
        <h4 className="font-semibold">Service Dependencies</h4>
        <div className="flex flex-wrap gap-2">
          {profile.startup_sequence.map((service: string, index: number) => (
            <div key={service} className="flex items-center space-x-2">
              <Badge variant="outline" className="flex items-center space-x-1">
                <span className="text-xs">{index + 1}</span>
                <span>{service}</span>
              </Badge>
              {index < profile.startup_sequence.length - 1 && (
                <ArrowRight className="h-3 w-3 text-gray-400" />
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className={cn('space-y-6', className)}>
      {/* Orchestration Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="h-5 w-5 text-blue-500" />
            <span>Service Orchestration</span>
          </CardTitle>
          <CardDescription>
            Configure and execute startup sequences with dependency management
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Tabs defaultValue="profile" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="profile">Profile-based</TabsTrigger>
              <TabsTrigger value="custom">Custom Selection</TabsTrigger>
            </TabsList>
            
            <TabsContent value="profile" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Startup Profile</label>
                  <Select value={selectedProfile} onValueChange={setSelectedProfile}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a startup profile" />
                    </SelectTrigger>
                    <SelectContent>
                      {profiles.map((profile) => (
                        <SelectItem key={profile.id} value={profile.id}>
                          <div className="flex items-center space-x-2">
                            <span>{profile.profile_name}</span>
                            {profile.is_default && (
                              <Badge variant="secondary" className="text-xs">Default</Badge>
                            )}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <label className="text-sm font-medium">Execution Mode</label>
                  <Select 
                    value={parallelExecution ? 'parallel' : 'sequential'} 
                    onValueChange={(value) => setParallelExecution(value === 'parallel')}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="sequential">
                        <div className="flex items-center space-x-2">
                          <ArrowRight className="h-4 w-4" />
                          <span>Sequential</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="parallel">
                        <div className="flex items-center space-x-2">
                          <GitBranch className="h-4 w-4" />
                          <span>Parallel</span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Timeout (seconds)</label>
                  <Select 
                    value={timeoutSeconds.toString()} 
                    onValueChange={(value) => setTimeoutSeconds(parseInt(value))}
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
                
                <div className="flex items-end">
                  <Button
                    onClick={() => setShowDependencyGraph(!showDependencyGraph)}
                    variant="outline"
                    className="w-full"
                  >
                    <GitBranch className="h-4 w-4 mr-2" />
                    {showDependencyGraph ? 'Hide' : 'Show'} Dependencies
                  </Button>
                </div>
              </div>

              {/* Dependency Graph */}
              {showDependencyGraph && selectedProfile && (
                <DependencyGraph
                  services={profiles.find(p => p.id === selectedProfile)?.startup_sequence || []}
                  dependencies={{}}
                  serviceStatuses={{}}
                  onServiceClick={(serviceName) => {
                    console.log('Service clicked:', serviceName);
                  }}
                />
              )}

              {/* Start Button */}
              <div className="flex space-x-2">
                <Button
                  onClick={handleStartSequence}
                  disabled={!selectedProfile || startupWithProfile.isPending}
                  className="flex-1"
                >
                  <Play className="h-4 w-4 mr-2" />
                  {startupWithProfile.isPending ? 'Starting...' : 'Start Sequence'}
                </Button>
                
                <Button variant="outline">
                  <Settings className="h-4 w-4 mr-2" />
                  Configure
                </Button>
              </div>
            </TabsContent>
            
            <TabsContent value="custom" className="space-y-4">
              <div className="text-center py-8 text-muted-foreground">
                <Users className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Custom service selection coming soon</p>
                <p className="text-sm">Select individual services and configure dependencies</p>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Active Sequences */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Timer className="h-5 w-5 text-green-500" />
              <span>Active Sequences</span>
            </div>
            <Badge variant="outline">
              {activeSequences.length} running
            </Badge>
          </CardTitle>
          <CardDescription>
            Monitor and control currently executing startup sequences
          </CardDescription>
        </CardHeader>
        <CardContent>
          {activeSequences.length > 0 ? (
            <div className="space-y-4">
              {activeSequences.map((sequence) => (
                <Card key={sequence.id} className="border-l-4 border-l-blue-500">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        {getSequenceStatusIcon(sequence.status)}
                        <h4 className="font-semibold">{sequence.sequence_name}</h4>
                        <Badge variant="outline">{sequence.status}</Badge>
                      </div>
                      
                      <div className="flex space-x-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleStopSequence(sequence.id)}
                          disabled={sequence.status !== SequenceStatus.RUNNING}
                        >
                          <Square className="h-3 w-3 mr-1" />
                          Stop
                        </Button>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Progress</span>
                        <span>{Math.round(getSequenceProgress(sequence))}%</span>
                      </div>
                      <Progress value={getSequenceProgress(sequence)} className="h-2" />
                    </div>
                    
                    <div className="mt-3 flex flex-wrap gap-1">
                      {sequence.services.map((service) => (
                        <Badge 
                          key={service} 
                          variant={sequence.services_completed?.includes(service) ? "default" : "secondary"}
                          className="text-xs"
                        >
                          {service}
                        </Badge>
                      ))}
                    </div>
                    
                    {sequence.started_at && (
                      <div className="mt-2 text-xs text-muted-foreground">
                        Started: {new Date(sequence.started_at).toLocaleString()}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Timer className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No active sequences</p>
              <p className="text-sm">Start a sequence to see it here</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default StartupOrchestrator;
