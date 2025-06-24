import { useState, useCallback } from 'react';

interface ModelPerformanceData {
  id: string;
  model_name: string;
  model_size_gb: number;
  usefulness_score: number;
  speed_rating: string;
  speed_seconds?: number;
  gpu_utilization: number;
  gpu_layers_offloaded: number;
  gpu_layers_total: number;
  context_window: number;
  parameters_billions: number;
  architecture: string;
  best_use_cases: string[];
  cost_per_token?: number;
  last_tested: string;
  test_results: Record<string, any>;
  user_ratings: Array<Record<string, any>>;
  performance_metrics: Record<string, any>;
  created_at: string;
  updated_at: string;
}

interface ModelStats {
  total_models: number;
  average_usefulness: number;
  best_model: {
    name: string;
    usefulness_score: number;
    architecture: string;
  };
  fastest_model: {
    name: string;
    speed_seconds: number;
    architecture: string;
  } | null;
  architectures: Record<string, {
    count: number;
    avg_usefulness: number;
  }>;
}

interface FetchModelsParams {
  page?: number;
  page_size?: number;
  architecture?: string;
  min_usefulness?: number;
  sort_by?: string;
  sort_order?: string;
}

interface ModelRecommendationRequest {
  task_type: string;
  priority: string;
  max_size_gb?: number;
  min_usefulness_score?: number;
}

interface ModelRecommendationResponse {
  recommended_models: ModelPerformanceData[];
  reasoning: string;
  task_type: string;
  priority: string;
}

export const useModelPerformance = () => {
  const [models, setModels] = useState<ModelPerformanceData[]>([]);
  const [stats, setStats] = useState<ModelStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = 'http://localhost:8000/api/v1';

  const fetchModels = useCallback(async (params: FetchModelsParams = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const queryParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          queryParams.append(key, value.toString());
        }
      });

      const response = await fetch(`${API_BASE}/models?${queryParams}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }

      const data = await response.json();
      setModels(data.models || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Error fetching models:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/models/stats/summary`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch stats: ${response.statusText}`);
      }

      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  }, []);

  const getModel = useCallback(async (modelName: string): Promise<ModelPerformanceData | null> => {
    try {
      const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelName)}`);
      
      if (!response.ok) {
        if (response.status === 404) {
          return null;
        }
        throw new Error(`Failed to fetch model: ${response.statusText}`);
      }

      return await response.json();
    } catch (err) {
      console.error('Error fetching model:', err);
      return null;
    }
  }, []);

  const getRecommendations = useCallback(async (
    request: ModelRecommendationRequest
  ): Promise<ModelRecommendationResponse | null> => {
    try {
      const response = await fetch(`${API_BASE}/models/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Failed to get recommendations: ${response.statusText}`);
      }

      return await response.json();
    } catch (err) {
      console.error('Error getting recommendations:', err);
      return null;
    }
  }, []);

  const rateModel = useCallback(async (
    modelName: string,
    rating: number,
    taskType: string,
    feedback?: string
  ): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelName)}/rate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: modelName,
          rating,
          task_type: taskType,
          feedback,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to rate model: ${response.statusText}`);
      }

      // Refresh models after rating
      await fetchModels();
      return true;
    } catch (err) {
      console.error('Error rating model:', err);
      return false;
    }
  }, [fetchModels]);

  const createOrUpdateModel = useCallback(async (
    modelData: Omit<ModelPerformanceData, 'id' | 'created_at' | 'updated_at' | 'last_tested' | 'user_ratings'>
  ): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE}/models/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(modelData),
      });

      if (!response.ok) {
        throw new Error(`Failed to create/update model: ${response.statusText}`);
      }

      // Refresh models after update
      await fetchModels();
      return true;
    } catch (err) {
      console.error('Error creating/updating model:', err);
      return false;
    }
  }, [fetchModels]);

  const deleteModel = useCallback(async (modelName: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelName)}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`Failed to delete model: ${response.statusText}`);
      }

      // Refresh models after deletion
      await fetchModels();
      return true;
    } catch (err) {
      console.error('Error deleting model:', err);
      return false;
    }
  }, [fetchModels]);

  return {
    models,
    stats,
    loading,
    error,
    fetchModels,
    fetchStats,
    getModel,
    getRecommendations,
    rateModel,
    createOrUpdateModel,
    deleteModel,
  };
};
