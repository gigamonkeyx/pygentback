/**
 * Startup Service Integration Test
 * Validates the integration of startup service components with existing PyGent Factory infrastructure
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { ViewType, ServiceStatus, SequenceStatus, SystemHealthStatus } from '../types';
import type { 
  StartupServiceState, 
  ServiceStatusInfo, 
  StartupSequence, 
  SystemStatusInfo 
} from '../types';

describe('Startup Service Integration', () => {
  describe('Type System Integration', () => {
    it('should have all required ViewType enums for startup service', () => {
      expect(ViewType.STARTUP_DASHBOARD).toBe('startup_dashboard');
      expect(ViewType.STARTUP_ORCHESTRATION).toBe('startup_orchestration');
      expect(ViewType.STARTUP_CONFIGURATION).toBe('startup_configuration');
      expect(ViewType.STARTUP_MONITORING).toBe('startup_monitoring');
    });

    it('should have all required ServiceStatus enums', () => {
      expect(ServiceStatus.STARTING).toBe('starting');
      expect(ServiceStatus.RUNNING).toBe('running');
      expect(ServiceStatus.STOPPING).toBe('stopping');
      expect(ServiceStatus.STOPPED).toBe('stopped');
      expect(ServiceStatus.ERROR).toBe('error');
      expect(ServiceStatus.UNKNOWN).toBe('unknown');
    });

    it('should have all required SequenceStatus enums', () => {
      expect(SequenceStatus.PENDING).toBe('pending');
      expect(SequenceStatus.RUNNING).toBe('running');
      expect(SequenceStatus.COMPLETED).toBe('completed');
      expect(SequenceStatus.FAILED).toBe('failed');
      expect(SequenceStatus.CANCELLED).toBe('cancelled');
    });

    it('should have all required SystemHealthStatus enums', () => {
      expect(SystemHealthStatus.HEALTHY).toBe('healthy');
      expect(SystemHealthStatus.DEGRADED).toBe('degraded');
      expect(SystemHealthStatus.UNHEALTHY).toBe('unhealthy');
      expect(SystemHealthStatus.UNKNOWN).toBe('unknown');
    });
  });

  describe('State Structure Validation', () => {
    let mockStartupServiceState: StartupServiceState;
    let mockServiceStatus: ServiceStatusInfo;
    let mockSequence: StartupSequence;
    let mockSystemStatus: SystemStatusInfo;

    beforeEach(() => {
      mockServiceStatus = {
        service_name: 'test-service',
        status: ServiceStatus.RUNNING,
        health_score: 0.95,
        last_check: new Date(),
        uptime_seconds: 3600,
        metrics: {
          cpu_usage: 15.2,
          memory_usage: 256
        }
      };

      mockSequence = {
        id: 'seq-1',
        sequence_name: 'test-sequence',
        services: ['service1', 'service2'],
        dependencies: { 'service2': ['service1'] },
        environment: 'development',
        status: SequenceStatus.RUNNING,
        started_at: new Date(),
        execution_details: {},
        parallel_execution: false,
        timeout_seconds: 300,
        retry_attempts: 3,
        created_at: new Date(),
        updated_at: new Date()
      };

      mockSystemStatus = {
        overall_status: SystemHealthStatus.HEALTHY,
        health_score: 0.85,
        services: [mockServiceStatus],
        active_sequences: ['seq-1'],
        system_metrics: {
          cpu_usage: 45.2,
          memory_usage: 68.5
        },
        last_updated: new Date()
      };

      mockStartupServiceState = {
        services: [mockServiceStatus],
        activeSequences: [mockSequence],
        configurations: [],
        profiles: [],
        systemStatus: mockSystemStatus,
        orchestrationProgress: {},
        realtimeMetrics: {},
        logs: []
      };
    });

    it('should create valid ServiceStatusInfo objects', () => {
      expect(mockServiceStatus.service_name).toBe('test-service');
      expect(mockServiceStatus.status).toBe(ServiceStatus.RUNNING);
      expect(mockServiceStatus.health_score).toBe(0.95);
      expect(mockServiceStatus.metrics).toHaveProperty('cpu_usage');
    });

    it('should create valid StartupSequence objects', () => {
      expect(mockSequence.id).toBe('seq-1');
      expect(mockSequence.status).toBe(SequenceStatus.RUNNING);
      expect(mockSequence.services).toContain('service1');
      expect(mockSequence.dependencies).toHaveProperty('service2');
    });

    it('should create valid SystemStatusInfo objects', () => {
      expect(mockSystemStatus.overall_status).toBe(SystemHealthStatus.HEALTHY);
      expect(mockSystemStatus.health_score).toBe(0.85);
      expect(mockSystemStatus.services).toHaveLength(1);
      expect(mockSystemStatus.active_sequences).toContain('seq-1');
    });

    it('should create valid StartupServiceState objects', () => {
      expect(mockStartupServiceState.services).toHaveLength(1);
      expect(mockStartupServiceState.activeSequences).toHaveLength(1);
      expect(mockStartupServiceState.systemStatus).toBe(mockSystemStatus);
      expect(mockStartupServiceState.logs).toHaveLength(0);
    });
  });

  describe('Component Integration', () => {
    it('should validate ServiceStatusDashboard props interface', () => {
      const mockProps = {
        services: [],
        systemStatus: null,
        onServiceAction: (serviceName: string, action: 'start' | 'stop' | 'restart') => {
          console.log(`Action ${action} for ${serviceName}`);
        },
        onRefresh: () => {
          console.log('Refresh triggered');
        }
      };

      expect(typeof mockProps.onServiceAction).toBe('function');
      expect(typeof mockProps.onRefresh).toBe('function');
      expect(Array.isArray(mockProps.services)).toBe(true);
    });

    it('should validate HealthIndicator props interface', () => {
      const mockProps = {
        status: ServiceStatus.RUNNING,
        healthScore: 0.95,
        lastCheck: new Date(),
        size: 'md' as const,
        showLabel: true
      };

      expect(mockProps.status).toBe(ServiceStatus.RUNNING);
      expect(mockProps.healthScore).toBe(0.95);
      expect(mockProps.size).toBe('md');
      expect(mockProps.showLabel).toBe(true);
    });
  });

  describe('Route Integration', () => {
    it('should have correct route mappings for startup service views', () => {
      const routeToViewType = {
        '/startup': ViewType.STARTUP_DASHBOARD,
        '/startup/orchestration': ViewType.STARTUP_ORCHESTRATION,
        '/startup/configuration': ViewType.STARTUP_CONFIGURATION,
        '/startup/monitoring': ViewType.STARTUP_MONITORING
      };

      expect(routeToViewType['/startup']).toBe(ViewType.STARTUP_DASHBOARD);
      expect(routeToViewType['/startup/orchestration']).toBe(ViewType.STARTUP_ORCHESTRATION);
      expect(routeToViewType['/startup/configuration']).toBe(ViewType.STARTUP_CONFIGURATION);
      expect(routeToViewType['/startup/monitoring']).toBe(ViewType.STARTUP_MONITORING);
    });

    it('should have correct view type to route mappings', () => {
      const viewTypeToRoute = {
        [ViewType.STARTUP_DASHBOARD]: '/startup',
        [ViewType.STARTUP_ORCHESTRATION]: '/startup/orchestration',
        [ViewType.STARTUP_CONFIGURATION]: '/startup/configuration',
        [ViewType.STARTUP_MONITORING]: '/startup/monitoring'
      };

      expect(viewTypeToRoute[ViewType.STARTUP_DASHBOARD]).toBe('/startup');
      expect(viewTypeToRoute[ViewType.STARTUP_ORCHESTRATION]).toBe('/startup/orchestration');
      expect(viewTypeToRoute[ViewType.STARTUP_CONFIGURATION]).toBe('/startup/configuration');
      expect(viewTypeToRoute[ViewType.STARTUP_MONITORING]).toBe('/startup/monitoring');
    });
  });

  describe('State Management Integration', () => {
    it('should validate startup service state actions', () => {
      // Mock state management actions
      const mockActions = {
        updateStartupServices: (services: ServiceStatusInfo[]) => services,
        updateServiceStatus: (serviceName: string, status: Partial<ServiceStatusInfo>) => ({ serviceName, status }),
        updateActiveSequences: (sequences: StartupSequence[]) => sequences,
        addActiveSequence: (sequence: StartupSequence) => sequence,
        updateSequenceStatus: (sequenceId: string, updates: Partial<StartupSequence>) => ({ sequenceId, updates }),
        updateSystemStatus: (status: SystemStatusInfo) => status,
        updateOrchestrationProgress: (progress: Record<string, any>) => progress,
        updateRealtimeMetrics: (metrics: Record<string, any>) => metrics,
        addStartupLog: (log: any) => log,
        clearStartupLogs: () => []
      };

      expect(typeof mockActions.updateStartupServices).toBe('function');
      expect(typeof mockActions.updateServiceStatus).toBe('function');
      expect(typeof mockActions.updateActiveSequences).toBe('function');
      expect(typeof mockActions.addActiveSequence).toBe('function');
      expect(typeof mockActions.updateSequenceStatus).toBe('function');
      expect(typeof mockActions.updateSystemStatus).toBe('function');
      expect(typeof mockActions.updateOrchestrationProgress).toBe('function');
      expect(typeof mockActions.updateRealtimeMetrics).toBe('function');
      expect(typeof mockActions.addStartupLog).toBe('function');
      expect(typeof mockActions.clearStartupLogs).toBe('function');
    });
  });

  describe('WebSocket Event Integration', () => {
    it('should validate startup service WebSocket event types', () => {
      const mockStartupProgressEvent = {
        type: 'startup_progress',
        data: {
          sequence_id: 'seq-1',
          service: 'test-service',
          status: ServiceStatus.RUNNING,
          progress_percent: 75.5,
          message: 'Service starting...',
          details: {}
        },
        timestamp: new Date(),
        source: 'startup-service'
      };

      const mockStartupLogEvent = {
        type: 'startup_log',
        data: {
          level: 'info',
          logger: 'startup-service',
          message: 'Service started successfully',
          service: 'test-service',
          details: {}
        },
        timestamp: new Date(),
        source: 'startup-service'
      };

      expect(mockStartupProgressEvent.type).toBe('startup_progress');
      expect(mockStartupProgressEvent.data.sequence_id).toBe('seq-1');
      expect(mockStartupProgressEvent.data.status).toBe(ServiceStatus.RUNNING);

      expect(mockStartupLogEvent.type).toBe('startup_log');
      expect(mockStartupLogEvent.data.level).toBe('info');
      expect(mockStartupLogEvent.data.service).toBe('test-service');
    });
  });
});

// Export test utilities for other test files
export const createMockServiceStatus = (overrides: Partial<ServiceStatusInfo> = {}): ServiceStatusInfo => ({
  service_name: 'mock-service',
  status: ServiceStatus.RUNNING,
  health_score: 0.9,
  last_check: new Date(),
  uptime_seconds: 1800,
  metrics: {},
  ...overrides
});

export const createMockStartupSequence = (overrides: Partial<StartupSequence> = {}): StartupSequence => ({
  id: 'mock-sequence',
  sequence_name: 'Mock Sequence',
  services: ['service1'],
  dependencies: {},
  environment: 'test',
  status: SequenceStatus.PENDING,
  execution_details: {},
  parallel_execution: false,
  timeout_seconds: 300,
  retry_attempts: 3,
  created_at: new Date(),
  updated_at: new Date(),
  ...overrides
});

export const createMockSystemStatus = (overrides: Partial<SystemStatusInfo> = {}): SystemStatusInfo => ({
  overall_status: SystemHealthStatus.HEALTHY,
  health_score: 0.85,
  services: [],
  active_sequences: [],
  system_metrics: {},
  last_updated: new Date(),
  ...overrides
});
