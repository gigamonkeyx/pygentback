import React, { useState } from 'react';
import { Server, Download, Trash2, RefreshCw, Activity, HardDrive, Cpu, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { useOllama } from '@/hooks/useOllama';
import { cn } from '@/utils/cn';

export const OllamaPage: React.FC = () => {
  const {
    models,
    serviceStatus,
    metrics,
    loading,
    error,
    fetchModels,
    pullModel,
    deleteModel,
    refreshStatus,
    refreshMetrics
  } = useOllama();

  const [newModelName, setNewModelName] = useState('');
  const [pullingModel, setPullingModel] = useState<string | null>(null);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);

  const handlePullModel = async () => {
    if (!newModelName.trim()) return;
    
    setPullingModel(newModelName);
    const success = await pullModel(newModelName);
    setPullingModel(null);
    
    if (success) {
      setNewModelName('');
    }
  };

  const handleDeleteModel = async (modelName: string) => {
    setDeletingModel(modelName);
    await deleteModel(modelName);
    setDeletingModel(null);
  };

  const handleRefresh = () => {
    fetchModels();
    refreshStatus();
    refreshMetrics();
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusColor = (isReady: boolean) => {
    return isReady ? 'bg-green-500' : 'bg-red-500';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Server className="h-8 w-8 text-blue-500" />
          <div>
            <h1 className="text-3xl font-bold">Ollama Management</h1>
            <p className="text-muted-foreground">
              Manage AI models and monitor service health
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button onClick={handleRefresh} variant="outline" disabled={loading}>
            <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4">
            <p className="text-red-800">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Service Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Service Status</CardTitle>
            <div className={cn("w-3 h-3 rounded-full", getStatusColor(serviceStatus?.is_ready || false))} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {serviceStatus?.is_ready ? 'Running' : 'Stopped'}
            </div>
            <p className="text-xs text-muted-foreground">
              {serviceStatus?.url || 'Not available'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Models</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics?.total_models || 0}</div>
            <p className="text-xs text-muted-foreground">
              {formatBytes(metrics?.total_size || 0)} total size
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatBytes(metrics?.memory_usage || 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              System memory
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">GPU Utilization</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics?.gpu_utilization ? `${metrics.gpu_utilization}%` : 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">
              Graphics processing
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Pull New Model */}
      <Card>
        <CardHeader>
          <CardTitle>Pull New Model</CardTitle>
          <CardDescription>
            Download a new model from the Ollama registry
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-2">
            <Input
              placeholder="Enter model name (e.g., llama3.1:8b, deepseek-coder:6.7b)"
              value={newModelName}
              onChange={(e) => setNewModelName(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handlePullModel()}
              disabled={!!pullingModel}
            />
            <Button 
              onClick={handlePullModel} 
              disabled={!newModelName.trim() || !!pullingModel}
            >
              {pullingModel ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Pulling...
                </>
              ) : (
                <>
                  <Download className="h-4 w-4 mr-2" />
                  Pull Model
                </>
              )}
            </Button>
          </div>
          {pullingModel && (
            <p className="text-sm text-muted-foreground mt-2">
              Pulling {pullingModel}... This may take several minutes.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Available Models */}
      <Card>
        <CardHeader>
          <CardTitle>Available Models</CardTitle>
          <CardDescription>
            Manage your downloaded AI models
          </CardDescription>
        </CardHeader>
        <CardContent>
          {models.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Server className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No models available</p>
              <p className="text-sm">Pull a model to get started</p>
            </div>
          ) : (
            <div className="space-y-3">
              {models.map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <h3 className="font-medium">{model.name}</h3>
                      <Badge variant="secondary">
                        {formatBytes(model.size)}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Modified: {new Date(model.modified_at).toLocaleDateString()}
                    </p>
                    {model.details && (
                      <p className="text-xs text-muted-foreground">
                        {model.details.family} • {model.details.parameter_size}
                      </p>
                    )}
                  </div>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() => handleDeleteModel(model.name)}
                    disabled={deletingModel === model.name}
                  >
                    {deletingModel === model.name ? (
                      <RefreshCw className="h-4 w-4 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
