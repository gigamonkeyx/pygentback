import React from 'react';
import { Brain, Settings, Play, Pause, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useReasoning } from '@/stores/appStore';

export const ReasoningPage: React.FC = () => {
  const { reasoningState, updateReasoningState, resetReasoningState } = useReasoning();

  const handleStart = () => {
    updateReasoningState({ isActive: true });
  };

  const handlePause = () => {
    updateReasoningState({ isActive: false });
  };

  const handleReset = () => {
    resetReasoningState();
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Brain className="h-8 w-8 text-blue-500" />
          <div>
            <h1 className="text-3xl font-bold">Tree of Thought Reasoning</h1>
            <p className="text-muted-foreground">
              Advanced multi-path reasoning with thought exploration
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="icon">
            <Settings className="h-4 w-4" />
          </Button>
          {reasoningState.isActive ? (
            <Button onClick={handlePause} variant="outline">
              <Pause className="h-4 w-4 mr-2" />
              Pause
            </Button>
          ) : (
            <Button onClick={handleStart}>
              <Play className="h-4 w-4 mr-2" />
              Start
            </Button>
          )}
          <Button onClick={handleReset} variant="outline">
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {reasoningState.isActive ? 'Active' : 'Idle'}
            </div>
            <p className="text-xs text-muted-foreground">
              Current reasoning state
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Processing Time</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {reasoningState.processingTime.toFixed(1)}s
            </div>
            <p className="text-xs text-muted-foreground">
              Total elapsed time
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(reasoningState.confidence * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              Current confidence score
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Paths Explored</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {reasoningState.pathsExplored}
            </div>
            <p className="text-xs text-muted-foreground">
              Total reasoning paths
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Thought Tree Visualization */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Thought Tree Visualization</CardTitle>
            <CardDescription>
              Interactive visualization of reasoning paths and thought exploration
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-96 flex items-center justify-center border-2 border-dashed border-muted-foreground/25 rounded-lg">
              <div className="text-center text-muted-foreground">
                <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium">Thought Tree Visualization</p>
                <p className="text-sm">
                  Interactive D3.js visualization will be implemented here
                </p>
                <p className="text-xs mt-2">
                  Features: Node expansion, path highlighting, confidence scoring
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Configuration Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Reasoning Configuration</CardTitle>
          <CardDescription>
            Configure reasoning parameters and strategies
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium">Reasoning Mode</label>
              <p className="text-sm text-muted-foreground mt-1">
                Current: {reasoningState.mode.replace('_', ' ').toUpperCase()}
              </p>
            </div>
            <div>
              <label className="text-sm font-medium">Task Complexity</label>
              <p className="text-sm text-muted-foreground mt-1">
                Current: {reasoningState.complexity.toUpperCase()}
              </p>
            </div>
            <div>
              <label className="text-sm font-medium">Thoughts Generated</label>
              <p className="text-sm text-muted-foreground mt-1">
                Total: {reasoningState.thoughts.length}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
