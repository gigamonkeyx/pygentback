import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Types
export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  read?: boolean;
  timestamp?: Date;
}

export interface SystemMetrics {
  cpu: number;
  memory: number;
  gpu?: number;
  network: {
    upload: number;
    download: number;
  };
  agents: {
    tot_reasoning: 'online' | 'offline' | 'error';
    rag_retrieval: 'online' | 'offline' | 'error';
  };
}

export interface UIState {
  sidebarOpen: boolean;
  activeView: 'chat' | 'reasoning' | 'monitoring' | 'mcp-marketplace' | 'settings';
  theme: 'light' | 'dark' | 'system';
  notifications: Notification[];
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  agent?: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface ReasoningState {
  isActive: boolean;
  thoughts: Array<{
    id: string;
    content: string;
    confidence: number;
    children: string[];
    parent?: string;
  }>;
  processingTime: number;
  confidence: number;
  pathsExplored: number;
}

interface AppState {
  // UI State
  ui: UIState;
  
  // System State
  systemMetrics: SystemMetrics | null;
  notifications: Notification[];
  
  // Chat State
  messages: ChatMessage[];
  
  // Reasoning State
  reasoningState: ReasoningState;
  
  // Actions
  toggleSidebar: () => void;
  setActiveView: (view: UIState['activeView']) => void;
  setTheme: (theme: UIState['theme']) => void;
  addNotification: (notification: Omit<Notification, 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  markNotificationRead: (id: string) => void;
  setSystemMetrics: (metrics: SystemMetrics) => void;
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  clearMessages: () => void;
  updateReasoningState: (state: Partial<ReasoningState>) => void;
}

const initialUIState: UIState = {
  sidebarOpen: true,
  activeView: 'chat',
  theme: 'system',
  notifications: []
};

const initialReasoningState: ReasoningState = {
  isActive: false,
  thoughts: [],
  processingTime: 0,
  confidence: 0,
  pathsExplored: 0
};

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        ui: initialUIState,
        systemMetrics: null,
        notifications: [],
        messages: [],
        reasoningState: initialReasoningState,

        // UI Actions
        toggleSidebar: () =>
          set((state) => ({
            ui: { ...state.ui, sidebarOpen: !state.ui.sidebarOpen }
          })),

        setActiveView: (view) =>
          set((state) => ({
            ui: { ...state.ui, activeView: view }
          })),

        setTheme: (theme) =>
          set((state) => ({
            ui: { ...state.ui, theme }
          })),

        // Notification Actions
        addNotification: (notification) =>
          set((state) => ({
            notifications: [
              ...state.notifications,
              {
                ...notification,
                timestamp: new Date(),
                read: false
              }
            ]
          })),

        removeNotification: (id) =>
          set((state) => ({
            notifications: state.notifications.filter(n => n.id !== id)
          })),

        markNotificationRead: (id) =>
          set((state) => ({
            notifications: state.notifications.map(n =>
              n.id === id ? { ...n, read: true } : n
            )
          })),

        // System Actions
        setSystemMetrics: (metrics) =>
          set({ systemMetrics: metrics }),

        // Chat Actions
        addMessage: (message) =>
          set((state) => ({
            messages: [
              ...state.messages,
              {
                ...message,
                id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                timestamp: new Date()
              }
            ]
          })),

        clearMessages: () =>
          set({ messages: [] }),

        // Reasoning Actions
        updateReasoningState: (newState) =>
          set((state) => ({
            reasoningState: { ...state.reasoningState, ...newState }
          }))
      }),
      {
        name: 'pygent-factory-store',
        partialize: (state) => ({
          ui: {
            sidebarOpen: state.ui.sidebarOpen,
            theme: state.ui.theme
          }
        })
      }
    ),
    {
      name: 'pygent-factory-store'
    }
  )
);