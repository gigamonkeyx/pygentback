/**
 * Startup Service API Client
 * API client for PyGent Factory Startup Service following existing service patterns
 */

import {
  ServiceStatusInfo,
  StartupSequence,
  ServiceConfiguration,
  ConfigurationProfile,
  SystemStatusInfo,
  StartupRequest,
  StartupResponse,
  ServiceStatus,
  SequenceStatus,
  APIResponse,
  PaginatedResponse
} from '@/types';

// API Configuration
const API_BASE_URL = import.meta.env.DEV 
  ? 'http://localhost:8000/api'
  : 'https://api.timpayne.net/api';

const STARTUP_API_PREFIX = '/startup';

// Request/Response Types
export interface StartupApiError {
  message: string;
  code?: string;
  details?: Record<string, any>;
}

export interface ServiceActionRequest {
  action: 'start' | 'stop' | 'restart';
  options?: Record<string, any>;
}

export interface SequenceActionRequest {
  action: 'start' | 'stop' | 'pause' | 'resume' | 'cancel';
  options?: Record<string, any>;
}

export interface CreateConfigurationRequest {
  service_name: string;
  service_type: string;
  configuration: Record<string, any>;
  environment: string;
  version?: string;
}

export interface UpdateConfigurationRequest {
  configuration?: Record<string, any>;
  environment?: string;
  version?: string;
  is_active?: boolean;
}

export interface CreateProfileRequest {
  profile_name: string;
  description?: string;
  profile_type: string;
  services_config: Record<string, any>;
  startup_sequence: string[];
  environment_variables?: Record<string, string>;
  tags?: string[];
}

export interface UpdateProfileRequest {
  profile_name?: string;
  description?: string;
  services_config?: Record<string, any>;
  startup_sequence?: string[];
  environment_variables?: Record<string, string>;
  is_default?: boolean;
  is_active?: boolean;
  tags?: string[];
}

/**
 * Startup Service API Client Class
 */
class StartupApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Generic API request method
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<APIResponse<T>> {
    const url = `${this.baseUrl}${STARTUP_API_PREFIX}${endpoint}`;
    
    const defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  // Service Status Methods
  async getServices(): Promise<APIResponse<ServiceStatusInfo[]>> {
    return this.request<ServiceStatusInfo[]>('/services');
  }

  async getService(serviceName: string): Promise<APIResponse<ServiceStatusInfo>> {
    return this.request<ServiceStatusInfo>(`/services/${encodeURIComponent(serviceName)}`);
  }

  async performServiceAction(
    serviceName: string, 
    request: ServiceActionRequest
  ): Promise<APIResponse<{ message: string; status: ServiceStatus }>> {
    return this.request(`/services/${encodeURIComponent(serviceName)}/action`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // System Status Methods
  async getSystemStatus(): Promise<APIResponse<SystemStatusInfo>> {
    return this.request<SystemStatusInfo>('/system/status');
  }

  async getSystemHealth(): Promise<APIResponse<{ health_score: number; status: string }>> {
    return this.request('/system/health');
  }

  // Startup Sequence Methods
  async getSequences(): Promise<APIResponse<StartupSequence[]>> {
    return this.request<StartupSequence[]>('/sequences');
  }

  async getSequence(sequenceId: string): Promise<APIResponse<StartupSequence>> {
    return this.request<StartupSequence>(`/sequences/${sequenceId}`);
  }

  async createSequence(sequence: Partial<StartupSequence>): Promise<APIResponse<StartupSequence>> {
    return this.request<StartupSequence>('/sequences', {
      method: 'POST',
      body: JSON.stringify(sequence),
    });
  }

  async updateSequence(
    sequenceId: string, 
    updates: Partial<StartupSequence>
  ): Promise<APIResponse<StartupSequence>> {
    return this.request<StartupSequence>(`/sequences/${sequenceId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async deleteSequence(sequenceId: string): Promise<APIResponse<{ message: string }>> {
    return this.request(`/sequences/${sequenceId}`, {
      method: 'DELETE',
    });
  }

  async performSequenceAction(
    sequenceId: string,
    request: SequenceActionRequest
  ): Promise<APIResponse<StartupResponse>> {
    return this.request<StartupResponse>(`/sequences/${sequenceId}/action`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Configuration Methods
  async getConfigurations(): Promise<APIResponse<ServiceConfiguration[]>> {
    return this.request<ServiceConfiguration[]>('/configurations');
  }

  async getConfiguration(configId: string): Promise<APIResponse<ServiceConfiguration>> {
    return this.request<ServiceConfiguration>(`/configurations/${configId}`);
  }

  async createConfiguration(
    request: CreateConfigurationRequest
  ): Promise<APIResponse<ServiceConfiguration>> {
    return this.request<ServiceConfiguration>('/configurations', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async updateConfiguration(
    configId: string,
    request: UpdateConfigurationRequest
  ): Promise<APIResponse<ServiceConfiguration>> {
    return this.request<ServiceConfiguration>(`/configurations/${configId}`, {
      method: 'PUT',
      body: JSON.stringify(request),
    });
  }

  async deleteConfiguration(configId: string): Promise<APIResponse<{ message: string }>> {
    return this.request(`/configurations/${configId}`, {
      method: 'DELETE',
    });
  }

  // Profile Methods
  async getProfiles(): Promise<APIResponse<ConfigurationProfile[]>> {
    return this.request<ConfigurationProfile[]>('/profiles');
  }

  async getProfile(profileId: string): Promise<APIResponse<ConfigurationProfile>> {
    return this.request<ConfigurationProfile>(`/profiles/${profileId}`);
  }

  async createProfile(request: CreateProfileRequest): Promise<APIResponse<ConfigurationProfile>> {
    return this.request<ConfigurationProfile>('/profiles', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async updateProfile(
    profileId: string,
    request: UpdateProfileRequest
  ): Promise<APIResponse<ConfigurationProfile>> {
    return this.request<ConfigurationProfile>(`/profiles/${profileId}`, {
      method: 'PUT',
      body: JSON.stringify(request),
    });
  }

  async deleteProfile(profileId: string): Promise<APIResponse<{ message: string }>> {
    return this.request(`/profiles/${profileId}`, {
      method: 'DELETE',
    });
  }

  async activateProfile(profileId: string): Promise<APIResponse<ConfigurationProfile>> {
    return this.request<ConfigurationProfile>(`/profiles/${profileId}/activate`, {
      method: 'POST',
    });
  }

  // Startup Operations
  async startupWithProfile(profileId: string, options: Partial<StartupRequest> = {}): Promise<APIResponse<StartupResponse>> {
    return this.request<StartupResponse>('/startup', {
      method: 'POST',
      body: JSON.stringify({
        profile_id: profileId,
        ...options,
      }),
    });
  }

  async startupWithServices(services: string[], options: Partial<StartupRequest> = {}): Promise<APIResponse<StartupResponse>> {
    return this.request<StartupResponse>('/startup', {
      method: 'POST',
      body: JSON.stringify({
        services,
        ...options,
      }),
    });
  }

  async emergencyStop(): Promise<APIResponse<{ message: string; stopped_services: string[] }>> {
    return this.request('/emergency-stop', {
      method: 'POST',
    });
  }

  // Monitoring and Logs
  async getLogs(params: {
    limit?: number;
    offset?: number;
    level?: string;
    service?: string;
    since?: string;
  } = {}): Promise<PaginatedResponse<any>> {
    const queryParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryParams.append(key, String(value));
      }
    });

    const endpoint = `/logs${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    return this.request<any[]>(endpoint);
  }

  async getMetrics(timeRange: string = '1h'): Promise<APIResponse<Record<string, any>>> {
    return this.request<Record<string, any>>(`/metrics?time_range=${timeRange}`);
  }

  // Health Check
  async healthCheck(): Promise<APIResponse<{ status: string; timestamp: string }>> {
    return this.request('/health');
  }
}

// Create singleton instance
export const startupApi = new StartupApiClient();

// Export the class for testing or custom instances
export { StartupApiClient };
