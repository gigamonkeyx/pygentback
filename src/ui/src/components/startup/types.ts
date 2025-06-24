/**
 * Startup Service Component Types
 * Type definitions for all startup service components
 */

import { 
  ServiceStatusInfo, 
  StartupSequence, 
  ServiceConfiguration, 
  ConfigurationProfile,
  SystemStatusInfo,
  StartupRequest,
  StartupLogEvent,
  ServiceStatus,
  SequenceStatus
} from '@/types';

// Dashboard Component Props
export interface ServiceStatusDashboardProps {
  services: ServiceStatusInfo[];
  systemStatus: SystemStatusInfo | null;
  onServiceAction: (serviceName: string, action: 'start' | 'stop' | 'restart') => void;
  onRefresh: () => void;
  className?: string;
}

export interface StartupOrchestratorProps {
  profiles: ConfigurationProfile[];
  activeSequences: StartupSequence[];
  onStartSequence: (request: StartupRequest) => void;
  onStopSequence: (sequenceId: string) => void;
  onCreateProfile: (profile: Partial<ConfigurationProfile>) => void;
  className?: string;
}

export interface SystemMetricsPanelProps {
  metrics: Record<string, any>;
  realtimeData: Record<string, any>;
  onMetricSelect: (metric: string) => void;
  refreshInterval?: number;
  className?: string;
}

// Configuration Component Props
export interface ConfigurationManagerProps {
  configurations: ServiceConfiguration[];
  profiles: ConfigurationProfile[];
  onSaveConfiguration: (config: Partial<ServiceConfiguration>) => void;
  onDeleteConfiguration: (configId: string) => void;
  onSaveProfile: (profile: Partial<ConfigurationProfile>) => void;
  onDeleteProfile: (profileId: string) => void;
  className?: string;
}

export interface ProfileManagerProps {
  profiles: ConfigurationProfile[];
  selectedProfile: ConfigurationProfile | null;
  onSelectProfile: (profile: ConfigurationProfile) => void;
  onCreateProfile: (profile: Partial<ConfigurationProfile>) => void;
  onUpdateProfile: (profileId: string, updates: Partial<ConfigurationProfile>) => void;
  onDeleteProfile: (profileId: string) => void;
  onDuplicateProfile: (profile: ConfigurationProfile) => void;
  className?: string;
}

export interface ServiceConfigEditorProps {
  configuration: ServiceConfiguration | null;
  onSave: (config: Partial<ServiceConfiguration>) => void;
  onCancel: () => void;
  onValidate: (config: Partial<ServiceConfiguration>) => Promise<string[]>;
  isEditing: boolean;
  className?: string;
}

// Monitoring Component Props
export interface LogViewerProps {
  logs: StartupLogEvent[];
  onClear: () => void;
  onFilter: (filters: LogFilter) => void;
  onExport: () => void;
  maxLines?: number;
  autoScroll?: boolean;
  className?: string;
}

export interface ProgressTrackerProps {
  steps: SequenceStep[];
  currentStep: number;
  progress?: number;
  status?: 'running' | 'completed' | 'failed' | 'paused';
  showStepDetails?: boolean;
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

export interface HealthIndicatorProps {
  status: ServiceStatus | SequenceStatus;
  healthScore?: number;
  lastCheck?: Date;
  errorMessage?: string;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  className?: string;
}

// Control Component Props
export interface ServiceControlsProps {
  serviceName: string;
  status: ServiceStatus;
  onStart?: () => void | Promise<void>;
  onStop?: () => void | Promise<void>;
  onRestart?: () => void | Promise<void>;
  onPause?: () => void | Promise<void>;
  onConfigure?: () => void | Promise<void>;
  onViewLogs?: () => void | Promise<void>;
  onViewDetails?: () => void | Promise<void>;
  disabled?: boolean;
  size?: 'sm' | 'default' | 'lg';
  variant?: 'default' | 'compact' | 'minimal';
  className?: string;
}

export interface SequenceControlsProps {
  sequenceName: string;
  status: SequenceStatus;
  progress?: number;
  services?: string[];
  completedServices?: string[];
  onStart?: () => void | Promise<void>;
  onStop?: () => void | Promise<void>;
  onPause?: () => void | Promise<void>;
  onResume?: () => void | Promise<void>;
  onRestart?: () => void | Promise<void>;
  onConfigure?: () => void | Promise<void>;
  executionMode?: 'sequential' | 'parallel';
  onExecutionModeChange?: (mode: 'sequential' | 'parallel') => void;
  timeoutSeconds?: number;
  onTimeoutChange?: (seconds: number) => void;
  disabled?: boolean;
  className?: string;
}

export interface QuickActionsProps {
  onStartAll?: () => void | Promise<void>;
  onStopAll?: () => void | Promise<void>;
  onRestartAll?: () => void | Promise<void>;
  onRefreshStatus?: () => void | Promise<void>;
  onOpenConfiguration?: () => void | Promise<void>;
  onOpenMonitoring?: () => void | Promise<void>;
  onRunTests?: () => void | Promise<void>;
  onExportLogs?: () => void | Promise<void>;
  onImportProfile?: () => void | Promise<void>;
  onCreateProfile?: () => void | Promise<void>;
  onOpenHelp?: () => void | Promise<void>;
  activeServices?: number;
  totalServices?: number;
  isLoading?: boolean;
  variant?: 'default' | 'compact' | 'minimal';
  className?: string;
}

// Utility Types
export interface LogFilter {
  level?: string[];
  service?: string[];
  timeRange?: {
    start: Date;
    end: Date;
  };
  searchTerm?: string;
}

export interface MetricConfig {
  key: string;
  label: string;
  unit?: string;
  format?: 'number' | 'percentage' | 'bytes' | 'duration';
  color?: string;
  threshold?: {
    warning: number;
    critical: number;
  };
}

export interface ChartData {
  timestamp: Date;
  value: number;
  label?: string;
}

export interface ServiceAction {
  id: string;
  label: string;
  icon: string;
  action: () => void;
  disabled?: boolean;
  destructive?: boolean;
}

export interface SequenceStep {
  id: string;
  name: string;
  description?: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  startTime?: Date;
  endTime?: Date;
  duration?: number;
  error?: string;
  dependencies?: string[];
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

// Event Handler Types
export type ServiceActionHandler = (serviceName: string, action: string) => void;
export type SequenceActionHandler = (sequenceId: string, action: string) => void;
export type ConfigurationSaveHandler = (config: Partial<ServiceConfiguration>) => void;
export type ProfileSaveHandler = (profile: Partial<ConfigurationProfile>) => void;
export type LogFilterHandler = (filters: LogFilter) => void;
export type MetricSelectHandler = (metric: string) => void;

// Component State Types
export interface DashboardState {
  selectedService: string | null;
  selectedSequence: string | null;
  viewMode: 'grid' | 'list' | 'detailed';
  refreshInterval: number;
  autoRefresh: boolean;
}

export interface OrchestratorState {
  selectedProfile: string | null;
  customServices: string[];
  advancedMode: boolean;
  parallelExecution: boolean;
  timeoutSeconds: number;
}

export interface ConfigurationState {
  selectedConfig: string | null;
  selectedProfile: string | null;
  editMode: boolean;
  validationErrors: string[];
  unsavedChanges: boolean;
}

export interface MonitoringState {
  logFilters: LogFilter;
  selectedMetrics: string[];
  timeRange: 'realtime' | '1h' | '6h' | '24h' | 'custom';
  autoScroll: boolean;
  showDetails: boolean;
}
