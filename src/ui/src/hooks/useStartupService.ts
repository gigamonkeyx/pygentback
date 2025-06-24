/**
 * Startup Service React Query Hooks
 * Custom hooks for startup service data fetching following existing patterns
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { startupApi, ServiceActionRequest, SequenceActionRequest } from '@/services/startupApi';
import { useStartupService as useStartupStore } from '@/stores/appStore';
import {
  ServiceStatusInfo,
  StartupSequence,
  ServiceConfiguration,
  ConfigurationProfile,
  SystemStatusInfo,
  StartupRequest,
  StartupResponse
} from '@/types';

// Query Keys
export const startupQueryKeys = {
  all: ['startup'] as const,
  services: () => [...startupQueryKeys.all, 'services'] as const,
  service: (name: string) => [...startupQueryKeys.services(), name] as const,
  sequences: () => [...startupQueryKeys.all, 'sequences'] as const,
  sequence: (id: string) => [...startupQueryKeys.sequences(), id] as const,
  configurations: () => [...startupQueryKeys.all, 'configurations'] as const,
  configuration: (id: string) => [...startupQueryKeys.configurations(), id] as const,
  profiles: () => [...startupQueryKeys.all, 'profiles'] as const,
  profile: (id: string) => [...startupQueryKeys.profiles(), id] as const,
  systemStatus: () => [...startupQueryKeys.all, 'system-status'] as const,
  systemHealth: () => [...startupQueryKeys.all, 'system-health'] as const,
  logs: (params?: any) => [...startupQueryKeys.all, 'logs', params] as const,
  metrics: (timeRange?: string) => [...startupQueryKeys.all, 'metrics', timeRange] as const,
};

// Service Status Hooks
export function useServices() {
  const { updateStartupServices } = useStartupStore();

  return useQuery({
    queryKey: startupQueryKeys.services(),
    queryFn: async () => {
      const response = await startupApi.getServices();
      if (response.success && response.data) {
        updateStartupServices(response.data);
      }
      return response.data || [];
    },
    refetchInterval: 5000, // Refresh every 5 seconds
    staleTime: 1000 * 30, // Consider stale after 30 seconds
  });
}

export function useService(serviceName: string) {
  const { updateServiceStatus } = useStartupStore();

  return useQuery({
    queryKey: startupQueryKeys.service(serviceName),
    queryFn: async () => {
      const response = await startupApi.getService(serviceName);
      if (response.success && response.data) {
        updateServiceStatus(serviceName, response.data);
      }
      return response.data;
    },
    enabled: !!serviceName,
    refetchInterval: 3000,
    staleTime: 1000 * 15,
  });
}

export function useServiceAction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ serviceName, action }: { serviceName: string; action: ServiceActionRequest }) => {
      return await startupApi.performServiceAction(serviceName, action);
    },
    onSuccess: (data, variables) => {
      // Invalidate and refetch service data
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.service(variables.serviceName) });
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.services() });
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.systemStatus() });
    },
  });
}

// System Status Hooks
export function useSystemStatus() {
  const { updateSystemStatus } = useStartupStore();

  return useQuery({
    queryKey: startupQueryKeys.systemStatus(),
    queryFn: async () => {
      const response = await startupApi.getSystemStatus();
      if (response.success && response.data) {
        updateSystemStatus(response.data);
      }
      return response.data;
    },
    refetchInterval: 10000, // Refresh every 10 seconds
    staleTime: 1000 * 60, // Consider stale after 1 minute
  });
}

export function useSystemHealth() {
  return useQuery({
    queryKey: startupQueryKeys.systemHealth(),
    queryFn: async () => {
      const response = await startupApi.getSystemHealth();
      return response.data;
    },
    refetchInterval: 15000,
    staleTime: 1000 * 45,
  });
}

// Startup Sequence Hooks
export function useSequences() {
  const { updateActiveSequences } = useStartupStore();

  return useQuery({
    queryKey: startupQueryKeys.sequences(),
    queryFn: async () => {
      const response = await startupApi.getSequences();
      if (response.success && response.data) {
        updateActiveSequences(response.data);
      }
      return response.data || [];
    },
    refetchInterval: 5000,
    staleTime: 1000 * 30,
  });
}

export function useSequence(sequenceId: string) {
  return useQuery({
    queryKey: startupQueryKeys.sequence(sequenceId),
    queryFn: async () => {
      const response = await startupApi.getSequence(sequenceId);
      return response.data;
    },
    enabled: !!sequenceId,
    refetchInterval: 3000,
    staleTime: 1000 * 15,
  });
}

export function useSequenceAction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ sequenceId, action }: { sequenceId: string; action: SequenceActionRequest }) => {
      return await startupApi.performSequenceAction(sequenceId, action);
    },
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.sequence(variables.sequenceId) });
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.sequences() });
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.systemStatus() });
    },
  });
}

// Configuration Hooks
export function useConfigurations() {
  const { updateConfigurations } = useStartupStore();

  return useQuery({
    queryKey: startupQueryKeys.configurations(),
    queryFn: async () => {
      const response = await startupApi.getConfigurations();
      if (response.success && response.data) {
        updateConfigurations(response.data);
      }
      return response.data || [];
    },
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

export function useConfiguration(configId: string) {
  return useQuery({
    queryKey: startupQueryKeys.configuration(configId),
    queryFn: async () => {
      const response = await startupApi.getConfiguration(configId);
      return response.data;
    },
    enabled: !!configId,
    staleTime: 1000 * 60 * 5,
  });
}

// Profile Hooks
export function useProfiles() {
  const { updateProfiles } = useStartupStore();

  return useQuery({
    queryKey: startupQueryKeys.profiles(),
    queryFn: async () => {
      const response = await startupApi.getProfiles();
      if (response.success && response.data) {
        updateProfiles(response.data);
      }
      return response.data || [];
    },
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

export function useProfile(profileId: string) {
  return useQuery({
    queryKey: startupQueryKeys.profile(profileId),
    queryFn: async () => {
      const response = await startupApi.getProfile(profileId);
      return response.data;
    },
    enabled: !!profileId,
    staleTime: 1000 * 60 * 5,
  });
}

// Startup Operations Hooks
export function useStartupWithProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ profileId, options }: { profileId: string; options?: Partial<StartupRequest> }) => {
      return await startupApi.startupWithProfile(profileId, options);
    },
    onSuccess: () => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.services() });
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.sequences() });
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.systemStatus() });
    },
  });
}

export function useStartupWithServices() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ services, options }: { services: string[]; options?: Partial<StartupRequest> }) => {
      return await startupApi.startupWithServices(services, options);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.services() });
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.sequences() });
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.systemStatus() });
    },
  });
}

export function useEmergencyStop() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      return await startupApi.emergencyStop();
    },
    onSuccess: () => {
      // Invalidate all startup-related queries
      queryClient.invalidateQueries({ queryKey: startupQueryKeys.all });
    },
  });
}

// Monitoring Hooks
export function useLogs(params?: any) {
  return useQuery({
    queryKey: startupQueryKeys.logs(params),
    queryFn: async () => {
      const response = await startupApi.getLogs(params);
      return response.data || [];
    },
    refetchInterval: 5000,
    staleTime: 1000 * 10,
  });
}

export function useMetrics(timeRange: string = '1h') {
  const { updateRealtimeMetrics } = useStartupStore();

  return useQuery({
    queryKey: startupQueryKeys.metrics(timeRange),
    queryFn: async () => {
      const response = await startupApi.getMetrics(timeRange);
      if (response.success && response.data) {
        updateRealtimeMetrics(response.data);
      }
      return response.data || {};
    },
    refetchInterval: 10000,
    staleTime: 1000 * 30,
  });
}

// Health Check Hook
export function useHealthCheck() {
  return useQuery({
    queryKey: [...startupQueryKeys.all, 'health'],
    queryFn: async () => {
      const response = await startupApi.healthCheck();
      return response.data;
    },
    refetchInterval: 30000, // Every 30 seconds
    staleTime: 1000 * 60, // 1 minute
    retry: 3,
  });
}
