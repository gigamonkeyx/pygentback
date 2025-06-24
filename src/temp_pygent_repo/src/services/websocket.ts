import { io, Socket } from 'socket.io-client';

export type EventHandler<T = any> = (data: T) => void;

interface WebSocketMessage {
  type: string;
  data: any;
}

class WebSocketService {
  private socket: Socket | null = null;
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
    // Set up default event handlers
    this.on('connect', () => {
      console.log('✅ WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connection_status', { connected: true });
    });

    this.on('disconnect', (reason: string) => {
      console.log('❌ WebSocket disconnected:', reason);
      this.emit('connection_status', { connected: false, reason });
      
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, don't reconnect
        return;
      }
      
      this.handleReconnect();
    });

    this.on('connect_error', (error: Error) => {
      console.error('❌ WebSocket connection error:', error);
      this.handleReconnect();
    });
  }

  async connect(url: string): Promise<boolean> {
    if (this.isConnecting) {
      return false;
    }

    this.isConnecting = true;
    this.url = url;

    return new Promise((resolve) => {
      try {
        // For development, use Socket.IO; for production, use native WebSocket
        if (import.meta.env.DEV) {
          this.socket = io(url.replace('/ws', ''), {
            transports: ['websocket'],
            upgrade: false,
            timeout: 10000,
            forceNew: true
          });
        } else {
          // For production, we'll use native WebSocket
          this.connectNativeWebSocket(url);
          resolve(true);
          return;
        }

        this.socket.on('connect', () => {
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          resolve(true);
        });

        this.socket.on('disconnect', (reason) => {
          this.isConnecting = false;
          this.emit('connection_status', { connected: false, reason });
        });

        this.socket.on('connect_error', (error) => {
          this.isConnecting = false;
          console.error('Socket.IO connection error:', error);
          resolve(false);
        });

        // Handle custom events
        this.socket.onAny((eventName, data) => {
          this.emit(eventName, data);
        });

      } catch (error) {
        this.isConnecting = false;
        console.error('Failed to create socket connection:', error);
        resolve(false);
      }
    });
  }

  private connectNativeWebSocket(url: string): void {
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        console.log('✅ Native WebSocket connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.emit('connection_status', { connected: true });
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.emit(message.type, message.data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        this.isConnecting = false;
        console.log('❌ Native WebSocket closed:', event.reason);
        this.emit('connection_status', { connected: false, reason: event.reason });
        
        if (event.code !== 1000) {
          this.handleReconnect();
        }
      };

      ws.onerror = (error) => {
        this.isConnecting = false;
        console.error('❌ Native WebSocket error:', error);
        this.handleReconnect();
      };

      // Store reference for sending messages
      (this as any).nativeWs = ws;

    } catch (error) {
      this.isConnecting = false;
      console.error('Failed to create native WebSocket:', error);
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
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    
    if ((this as any).nativeWs) {
      (this as any).nativeWs.close(1000, 'Client disconnect');
      (this as any).nativeWs = null;
    }
    
    this.eventHandlers.clear();
    this.reconnectAttempts = 0;
    this.isConnecting = false;
  }

  on<T = any>(event: string, handler: EventHandler<T>): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
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
    if (this.socket && this.socket.connected) {
      this.socket.emit(type, data);
    } else if ((this as any).nativeWs && (this as any).nativeWs.readyState === WebSocket.OPEN) {
      const message: WebSocketMessage = { type, data };
      (this as any).nativeWs.send(JSON.stringify(message));
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

  isConnected(): boolean {
    if (this.socket) {
      return this.socket.connected;
    }
    if ((this as any).nativeWs) {
      return (this as any).nativeWs.readyState === WebSocket.OPEN;
    }
    return false;
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

// Export types for use in components
export type { EventHandler };