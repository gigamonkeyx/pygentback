import {
  WebSocketEvent,
  ChatEvent,
  ReasoningEvent,
  EvolutionEvent,
  SystemEvent,
  ChatMessage,
  ReasoningState,
  EvolutionState,
  SystemMetrics,
  StartupProgressEvent,
  StartupSystemMetricsEvent,
  StartupLogEvent,
  ServiceStatusInfo,
  SystemStatusInfo
} from '@/types';

export type EventHandler<T = any> = (data: T) => void;

// WebSocket message format (matches backend)
interface WebSocketMessage {
  type: string;
  data: any;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private eventHandlers: Map<string, EventHandler[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private url = '';

  constructor() {
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    // Set up default event handlers for native WebSocket events
    this.on('connection_status', (data: { connected: boolean; reason?: string }) => {
      if (data.connected) {
        console.log('✅ WebSocket connected');
        this.reconnectAttempts = 0;
      } else {
        console.log('❌ WebSocket disconnected:', data.reason || 'Unknown reason');
      }
    });

    this.on('connection_error', (data: { error: string }) => {
      console.error('❌ WebSocket connection error:', data.error);
    });
  }

  connect(url: string = '/'): Promise<boolean> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        console.log('WebSocket already connected');
        resolve(true);
        return;
      }

      if (this.isConnecting) {
        console.log('WebSocket connection already in progress, waiting...');
        // Wait for current connection attempt
        const checkConnection = () => {
          if (this.ws?.readyState === WebSocket.OPEN) {
            resolve(true);
          } else if (!this.isConnecting) {
            reject(new Error('Connection failed'));
          } else {
            setTimeout(checkConnection, 100);
          }
        };
        checkConnection();
        return;
      }

      this.isConnecting = true;

      // Determine WebSocket URL - always connect to backend on port 8000
      let wsUrl: string;
      if (url.startsWith('ws')) {
        wsUrl = url;
      } else {
        // For development, connect directly to backend on port 8000
        // The Vite proxy doesn't always work reliably with WebSockets
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        if (isDevelopment) {
          wsUrl = `${protocol}//localhost:8000/ws`;
        } else {
          wsUrl = `${protocol}//${window.location.host}/ws`;
        }
      }
      this.url = wsUrl;

      try {
        console.log('Attempting WebSocket connection to:', wsUrl);
        this.ws = new WebSocket(wsUrl);

        // Set up connection timeout
        const connectionTimeout = setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            console.error('WebSocket connection timeout');
            this.isConnecting = false;
            if (this.ws) {
              this.ws.close();
            }
            reject(new Error('WebSocket connection timeout'));
          }
        }, 5000);

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          console.log('✅ WebSocket connected successfully to:', wsUrl);
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.emit('connection_status', { connected: true });

          // Send a ping to test the connection
          this.sendMessage('ping', { timestamp: new Date().toISOString() });

          resolve(true);
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          console.log('WebSocket disconnected:', event.reason || 'connection closed');
          this.isConnecting = false;
          this.emit('connection_status', { connected: false, reason: event.reason });

          // Auto-reconnect unless it was a clean close
          if (event.code !== 1000) {
            this.handleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          console.error('WebSocket error occurred:', error);
          this.isConnecting = false;
          this.emit('connection_error', { error: 'WebSocket connection failed' });
          // Don't reject immediately, let the timeout handle it
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  private handleMessage(message: WebSocketMessage): void {
    // Emit the event to registered handlers
    this.emit(message.type, message.data);

    // Handle specific message types
    switch (message.type) {
      case 'pong':
        console.log('Received pong from server');
        break;
      case 'chat_response':
        this.emit('chat_message', message.data);
        break;
      case 'reasoning_update':
        this.emit('reasoning_state', message.data);
        break;
      case 'evolution_update':
        this.emit('evolution_state', message.data);
        break;
      case 'system_metrics':
        this.emit('system_update', message.data);
        break;
      // Startup Service Events
      case 'startup_progress':
        this.emit('startup_progress_update', message.data);
        break;
      case 'startup_system_metrics':
        this.emit('startup_metrics_update', message.data);
        break;
      case 'startup_log':
        this.emit('startup_log_message', message.data);
        break;
      case 'startup_service_status':
        this.emit('startup_service_status_update', message.data);
        break;
      case 'startup_sequence_status':
        this.emit('startup_sequence_status_update', message.data);
        break;
      default:
        // Let other handlers process unknown message types
        break;
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);
    
    setTimeout(() => {
      this.connect(this.url);
    }, delay);
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.eventHandlers.clear();
    this.reconnectAttempts = 0;
    this.isConnecting = false;
  }

  on<T = any>(event: string, handler: EventHandler<T>): () => void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
    
    // Return unsubscribe function
    return () => this.off(event, handler);
  }

  off(event: string, handler?: EventHandler): void {
    if (!this.eventHandlers.has(event)) {
      return;
    }

    if (handler) {
      const handlers = this.eventHandlers.get(event)!;
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    } else {
      this.eventHandlers.delete(event);
    }
  }

  emit(event: string, data?: any): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in event handler for ${event}:`, error);
        }
      });
    }
  }

  sendMessage(type: string, data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const message: WebSocketMessage = { type, data };
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('Cannot send message: WebSocket not connected');
    }
  }

  // Convenience methods for common operations
  sendChatMessage(message: string, agent?: string): void {
    this.sendMessage('chat_message', {
      content: message,
      agent: agent || 'general',
      timestamp: new Date().toISOString()
    });
  }

  startReasoning(problem: string, mode: string = 'adaptive'): void {
    this.sendMessage('start_reasoning', {
      problem,
      mode,
      timestamp: new Date().toISOString()
    });
  }

  stopReasoning(): void {
    this.sendMessage('stop_reasoning', {
      timestamp: new Date().toISOString()
    });
  }

  requestSystemMetrics(): void {
    this.sendMessage('request_system_metrics', {
      timestamp: new Date().toISOString()
    });
  }

  // Startup Service Methods
  requestStartupStatus(): void {
    this.sendMessage('request_startup_status', {
      timestamp: new Date().toISOString()
    });
  }

  startService(serviceName: string, options: any = {}): void {
    this.sendMessage('start_service', {
      service_name: serviceName,
      options,
      timestamp: new Date().toISOString()
    });
  }

  stopService(serviceName: string, options: any = {}): void {
    this.sendMessage('stop_service', {
      service_name: serviceName,
      options,
      timestamp: new Date().toISOString()
    });
  }

  restartService(serviceName: string, options: any = {}): void {
    this.sendMessage('restart_service', {
      service_name: serviceName,
      options,
      timestamp: new Date().toISOString()
    });
  }

  startSequence(sequenceId: string, options: any = {}): void {
    this.sendMessage('start_sequence', {
      sequence_id: sequenceId,
      options,
      timestamp: new Date().toISOString()
    });
  }

  stopSequence(sequenceId: string): void {
    this.sendMessage('stop_sequence', {
      sequence_id: sequenceId,
      timestamp: new Date().toISOString()
    });
  }

  subscribeToStartupEvents(): void {
    this.sendMessage('subscribe_startup_events', {
      events: ['startup_progress', 'startup_system_metrics', 'startup_log', 'startup_service_status'],
      timestamp: new Date().toISOString()
    });
  }

  unsubscribeFromStartupEvents(): void {
    this.sendMessage('unsubscribe_startup_events', {
      timestamp: new Date().toISOString()
    });
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN || false;
  }

  getConnectionState(): {
    connected: boolean;
    reconnectAttempts: number;
    isConnecting: boolean;
  } {
    return {
      connected: this.isConnected(),
      reconnectAttempts: this.reconnectAttempts,
      isConnecting: this.isConnecting
    };
  }
}

// Create singleton instance
export const websocketService = new WebSocketService();