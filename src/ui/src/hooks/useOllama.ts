import { useState, useEffect, useCallback } from 'react';
import { OllamaModel, OllamaServiceStatus, OllamaMetrics } from '@/types';
import { websocketService } from '@/services/websocket';

interface UseOllamaReturn {
  // State
  models: OllamaModel[];
  serviceStatus: OllamaServiceStatus | null;
  metrics: OllamaMetrics | null;
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchModels: () => Promise<void>;
  pullModel: (modelName: string) => Promise<boolean>;
  deleteModel: (modelName: string) => Promise<boolean>;
  refreshStatus: () => Promise<void>;
  refreshMetrics: () => Promise<void>;
}

export const useOllama = (): UseOllamaReturn => {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [serviceStatus, setServiceStatus] = useState<OllamaServiceStatus | null>(null);
  const [metrics, setMetrics] = useState<OllamaMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = 'http://localhost:8080/api/v1';

  // Fetch available models
  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/ollama/models`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }

      const data = await response.json();
      setModels(data.models || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Error fetching Ollama models:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Pull a new model
  const pullModel = useCallback(async (modelName: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/ollama/models/pull`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName }),
      });

      if (!response.ok) {
        throw new Error(`Failed to pull model: ${response.statusText}`);
      }

      // Refresh models list after successful pull
      await fetchModels();
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Error pulling Ollama model:', err);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchModels]);

  // Delete a model
  const deleteModel = useCallback(async (modelName: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/ollama/models/${encodeURIComponent(modelName)}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`Failed to delete model: ${response.statusText}`);
      }

      // Refresh models list after successful deletion
      await fetchModels();
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Error deleting Ollama model:', err);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchModels]);

  // Refresh service status
  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/ollama/status`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch status: ${response.statusText}`);
      }

      const data = await response.json();
      setServiceStatus(data);
    } catch (err) {
      console.error('Error fetching Ollama status:', err);
      setServiceStatus(null);
    }
  }, []);

  // Refresh metrics
  const refreshMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/ollama/metrics`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch metrics: ${response.statusText}`);
      }

      const data = await response.json();
      setMetrics({
        ...data,
        last_updated: new Date()
      });
    } catch (err) {
      console.error('Error fetching Ollama metrics:', err);
    }
  }, []);

  // Set up WebSocket listeners for real-time updates
  useEffect(() => {
    const handleOllamaStatus = (data: any) => {
      if (data.status) {
        setServiceStatus(data.status);
      }
    };

    const handleOllamaModelUpdate = (data: any) => {
      if (data.models) {
        setModels(data.models);
      }
    };

    const handleOllamaMetrics = (data: any) => {
      if (data.metrics) {
        setMetrics({
          ...data.metrics,
          last_updated: new Date()
        });
      }
    };

    const handleOllamaError = (data: any) => {
      if (data.error) {
        setError(data.error);
      }
    };

    // Subscribe to WebSocket events
    const unsubscribeStatus = websocketService.on('ollama_status', handleOllamaStatus);
    const unsubscribeModels = websocketService.on('ollama_model_update', handleOllamaModelUpdate);
    const unsubscribeMetrics = websocketService.on('ollama_metrics', handleOllamaMetrics);
    const unsubscribeError = websocketService.on('ollama_error', handleOllamaError);

    return () => {
      unsubscribeStatus();
      unsubscribeModels();
      unsubscribeMetrics();
      unsubscribeError();
    };
  }, []);

  // Initial data fetch
  useEffect(() => {
    fetchModels();
    refreshStatus();
    refreshMetrics();
  }, [fetchModels, refreshStatus, refreshMetrics]);

  return {
    models,
    serviceStatus,
    metrics,
    loading,
    error,
    fetchModels,
    pullModel,
    deleteModel,
    refreshStatus,
    refreshMetrics,
  };
};
