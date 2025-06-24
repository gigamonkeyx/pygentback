import React, { useState, useEffect } from 'react';
import { Brain, Play, Square, BarChart3, TreePine } from 'lucide-react';
import { useAppStore } from '@/stores/appStore';
import { websocketService } from '@/services/websocket';

export const ReasoningPage: React.FC = () => {
  const [problem, setProblem] = useState('');
  const [mode, setMode] = useState('adaptive');
  const { reasoningState, updateReasoningState } = useAppStore();

  const modes = [
    { id: 'adaptive', name: 'Adaptive', description: 'Dynamically adjusts reasoning strategy' },
    { id: 'breadth_first', name: 'Breadth First', description: 'Explores all possibilities at each level' },
    { id: 'depth_first', name: 'Depth First', description: 'Explores deeply before backtracking' },
    { id: 'best_first', name: 'Best First', description: 'Follows most promising paths' }
  ];

  useEffect(() => {
    // Set up WebSocket event handlers for reasoning
    websocketService.on('reasoning_update', (data) => {
      updateReasoningState({
        thoughts: data.thoughts || reasoningState.thoughts,
        confidence: data.confidence || reasoningState.confidence,
        pathsExplored: data.pathsExplored || reasoningState.pathsExplored,
        processingTime: data.processingTime || reasoningState.processingTime
      });
    });

    websocketService.on('reasoning_complete', (data) => {
      updateReasoningState({
        isActive: false,
        thoughts: data.thoughts || reasoningState.thoughts,
        confidence: data.confidence || reasoningState.confidence,
        pathsExplored: data.pathsExplored || reasoningState.pathsExplored,
        processingTime: data.processingTime || reasoningState.processingTime
      });
    });

    return () => {
      websocketService.off('reasoning_update');
      websocketService.off('reasoning_complete');
    };
  }, [updateReasoningState, reasoningState]);

  const handleStart = () => {
    if (!problem.trim()) return;
    
    updateReasoningState({ isActive: true, thoughts: [], confidence: 0, pathsExplored: 0 });
    websocketService.startReasoning(problem, mode);
  };

  const handleStop = () => {
    updateReasoningState({ isActive: false });
    websocketService.stopReasoning();
  };

  const renderThoughtTree = () => {
    if (reasoningState.thoughts.length === 0) {
      return (
        <div className="text-center text-muted-foreground py-8">
          <TreePine className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No reasoning tree available</p>
          <p className="text-sm">Start reasoning to see the thought process</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {reasoningState.thoughts.map((thought) => (
          <div
            key={thought.id}
            className="p-4 border border-border rounded-lg bg-card hover:shadow-md transition-shadow"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <p className="text-sm text-foreground">{thought.content}</p>
                <div className="mt-2 flex items-center space-x-4 text-xs text-muted-foreground">
                  <span>Confidence: {(thought.confidence * 100).toFixed(1)}%</span>
                  <span>Children: {thought.children.length}</span>
                </div>
              </div>
              <div className="ml-4">
                <div className={`w-3 h-3 rounded-full ${
                  thought.confidence > 0.7 ? 'bg-green-500' :
                  thought.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-3">
        <Brain className="w-8 h-8 text-primary" />
        <div>
          <h1 className="text-2xl font-bold text-foreground">Tree of Thought Reasoning</h1>
          <p className="text-muted-foreground">Advanced AI reasoning with thought exploration</p>
        </div>
      </div>

      {/* Control Panel */}
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="space-y-4">
          {/* Problem Input */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Problem to Solve
            </label>
            <textarea
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              placeholder="Describe the problem you want the AI to reason about..."
              className="w-full p-3 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
              rows={3}
              disabled={reasoningState.isActive}
            />
          </div>

          {/* Mode Selection */}
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Reasoning Mode
            </label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="w-full p-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
              disabled={reasoningState.isActive}
            >
              {modes.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} - {m.description}
                </option>
              ))}
            </select>
          </div>

          {/* Controls */}
          <div className="flex items-center space-x-4">
            {!reasoningState.isActive ? (
              <button
                onClick={handleStart}
                disabled={!problem.trim()}
                className="flex items-center space-x-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Play className="w-4 h-4" />
                <span>Start Reasoning</span>
              </button>
            ) : (
              <button
                onClick={handleStop}
                className="flex items-center space-x-2 px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 transition-colors"
              >
                <Square className="w-4 h-4" />
                <span>Stop Reasoning</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-primary" />
            <span className="text-sm font-medium text-foreground">Confidence</span>
          </div>
          <p className="text-2xl font-bold text-foreground mt-2">
            {(reasoningState.confidence * 100).toFixed(1)}%
          </p>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <TreePine className="w-5 h-5 text-primary" />
            <span className="text-sm font-medium text-foreground">Thoughts</span>
          </div>
          <p className="text-2xl font-bold text-foreground mt-2">
            {reasoningState.thoughts.length}
          </p>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Brain className="w-5 h-5 text-primary" />
            <span className="text-sm font-medium text-foreground">Paths Explored</span>
          </div>
          <p className="text-2xl font-bold text-foreground mt-2">
            {reasoningState.pathsExplored}
          </p>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-primary" />
            <span className="text-sm font-medium text-foreground">Time (s)</span>
          </div>
          <p className="text-2xl font-bold text-foreground mt-2">
            {(reasoningState.processingTime / 1000).toFixed(1)}
          </p>
        </div>
      </div>

      {/* Thought Tree */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">Reasoning Tree</h2>
        <div className="max-h-96 overflow-y-auto">
          {renderThoughtTree()}
        </div>
      </div>
    </div>
  );
};