import React, { useEffect, useState } from 'react';
import { X, Brain, GitBranch, Target, Clock, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useReasoning } from '@/stores/appStore';
import { websocketService } from '@/services/websocket';
import { ThoughtNode, ReasoningMode, TaskComplexity } from '@/types';
import { cn } from '@/utils/cn';

interface ReasoningPanelProps {
  onClose: () => void;
  className?: string;
}

export const ReasoningPanel: React.FC<ReasoningPanelProps> = ({
  onClose,
  className
}) => {
  const { reasoningState, updateReasoningState } = useReasoning();
  const [selectedNode, setSelectedNode] = useState<ThoughtNode | null>(null);

  useEffect(() => {
    // Subscribe to reasoning updates
    const unsubscribeUpdate = websocketService.on('reasoning_update', (data: any) => {
      if (data.state) {
        updateReasoningState(data.state);
      }
      if (data.thought) {
        // Add new thought to the state
        const newThoughts = [...reasoningState.thoughts, data.thought];
        updateReasoningState({ thoughts: newThoughts });
      }
    });

    const unsubscribeComplete = websocketService.on('reasoning_complete', (data: any) => {
      if (data.state) {
        updateReasoningState({ ...data.state, isActive: false });
      }
    });

    return () => {
      unsubscribeUpdate();
      unsubscribeComplete();
    };
  }, [reasoningState.thoughts, updateReasoningState]);

  const getReasoningModeColor = (mode: ReasoningMode) => {
    switch (mode) {
      case ReasoningMode.TOT_ONLY:
        return 'text-blue-500';
      case ReasoningMode.RAG_ONLY:
        return 'text-green-500';
      case ReasoningMode.S3_RAG:
        return 'text-purple-500';
      case ReasoningMode.TOT_ENHANCED_RAG:
        return 'text-orange-500';
      case ReasoningMode.ADAPTIVE:
        return 'text-indigo-500';
      default:
        return 'text-gray-500';
    }
  };

  const getComplexityColor = (complexity: TaskComplexity) => {
    switch (complexity) {
      case TaskComplexity.SIMPLE:
        return 'text-green-500';
      case TaskComplexity.MODERATE:
        return 'text-yellow-500';
      case TaskComplexity.COMPLEX:
        return 'text-orange-500';
      case TaskComplexity.RESEARCH:
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const renderThoughtTree = () => {
    if (reasoningState.thoughts.length === 0) {
      return (
        <div className="flex items-center justify-center h-32 text-muted-foreground">
          <div className="text-center">
            <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No reasoning paths yet</p>
          </div>
        </div>
      );
    }

    // Group thoughts by depth for tree visualization
    const thoughtsByDepth = reasoningState.thoughts.reduce((acc, thought) => {
      if (!acc[thought.depth]) {
        acc[thought.depth] = [];
      }
      acc[thought.depth].push(thought);
      return acc;
    }, {} as Record<number, ThoughtNode[]>);

    return (
      <div className="space-y-3">
        {Object.entries(thoughtsByDepth).map(([depth, thoughts]) => (
          <div key={depth} className="space-y-2">
            <div className="flex items-center space-x-2 text-xs text-muted-foreground">
              <GitBranch className="h-3 w-3" />
              <span>Depth {depth}</span>
            </div>
            <div className="space-y-2 ml-4">
              {thoughts.map((thought) => (
                <Card
                  key={thought.id}
                  className={cn(
                    'cursor-pointer transition-colors hover:bg-accent',
                    selectedNode?.id === thought.id && 'ring-2 ring-primary'
                  )}
                  onClick={() => setSelectedNode(thought)}
                >
                  <CardContent className="p-3">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <div
                          className={cn(
                            'w-2 h-2 rounded-full',
                            thought.value_score > 0.8 ? 'bg-green-500' :
                            thought.value_score > 0.6 ? 'bg-yellow-500' :
                            'bg-red-500'
                          )}
                        />
                        <span className="text-xs font-medium">
                          Score: {thought.value_score.toFixed(2)}
                        </span>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {thought.confidence.toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-sm text-foreground line-clamp-2">
                      {thought.content}
                    </p>
                    {thought.reasoning_step && (
                      <p className="text-xs text-muted-foreground mt-1">
                        {thought.reasoning_step}
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderSelectedThought = () => {
    if (!selectedNode) {
      return (
        <div className="flex items-center justify-center h-32 text-muted-foreground">
          <div className="text-center">
            <Target className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Select a thought to view details</p>
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="font-medium">Thought Details</h4>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelectedNode(null)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="space-y-3">
          <div>
            <label className="text-xs font-medium text-muted-foreground">Content</label>
            <p className="text-sm mt-1">{selectedNode.content}</p>
          </div>
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-medium text-muted-foreground">Value Score</label>
              <p className="text-sm mt-1 font-mono">{selectedNode.value_score.toFixed(3)}</p>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground">Confidence</label>
              <p className="text-sm mt-1 font-mono">{selectedNode.confidence.toFixed(1)}%</p>
            </div>
          </div>
          
          <div>
            <label className="text-xs font-medium text-muted-foreground">Reasoning Step</label>
            <p className="text-sm mt-1">{selectedNode.reasoning_step || 'N/A'}</p>
          </div>
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-medium text-muted-foreground">Depth</label>
              <p className="text-sm mt-1 font-mono">{selectedNode.depth}</p>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground">Children</label>
              <p className="text-sm mt-1 font-mono">{selectedNode.children.length}</p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={cn('flex flex-col h-full bg-background', className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-2">
          <Brain className="h-5 w-5 text-blue-500" />
          <h3 className="font-semibold">Reasoning Panel</h3>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Status */}
      <div className="p-4 border-b">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs font-medium text-muted-foreground">Mode</label>
            <p className={cn('text-sm font-medium', getReasoningModeColor(reasoningState.mode))}>
              {reasoningState.mode.replace('_', ' ').toUpperCase()}
            </p>
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground">Complexity</label>
            <p className={cn('text-sm font-medium', getComplexityColor(reasoningState.complexity))}>
              {reasoningState.complexity.toUpperCase()}
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-4 mt-3">
          <div className="flex items-center space-x-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">Time</p>
              <p className="text-sm font-mono">{reasoningState.processingTime.toFixed(1)}s</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">Confidence</p>
              <p className="text-sm font-mono">{(reasoningState.confidence * 100).toFixed(1)}%</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <GitBranch className="h-4 w-4 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">Paths</p>
              <p className="text-sm font-mono">{reasoningState.pathsExplored}</p>
            </div>
          </div>
        </div>

        {reasoningState.isActive && (
          <div className="mt-3 flex items-center space-x-2 text-sm text-blue-500">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
            <span>Reasoning in progress...</span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full flex flex-col">
          {/* Thought Tree */}
          <div className="flex-1 p-4 overflow-y-auto">
            <h4 className="font-medium mb-3">Thought Tree</h4>
            {renderThoughtTree()}
          </div>

          {/* Selected Thought Details */}
          {selectedNode && (
            <div className="border-t p-4">
              {renderSelectedThought()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
