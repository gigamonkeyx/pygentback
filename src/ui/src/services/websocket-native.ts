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

  connect(url: string = '/'): Promise<boolean> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve(true);
        return;
      }

      if (this.isConnecting) {
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

      this.isConnecting = true;      // Determine WebSocket URL - always connect to backend on port 8000
      let wsUrl: string;
      if (url.startsWith('ws')) {
        wsUrl = url;
      } else {
        // For development, connect to backend on port 8000
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsUrl = `${protocol}//localhost:8000/ws`;
      }
      this.url = wsUrl;

      try {
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connected to:', wsUrl);
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.emit('connection_status', { connected: true });

          // Send a ping to test the connection
          this.sendMessage('ping', { timestamp: new Date().toISOString() });

          resolve(true);
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.reason || 'connection closed');
          this.isConnecting = false;
          this.emit('connection_status', { connected: false, reason: event.reason });

          // Auto-reconnect unless it was a clean close
          if (event.code !== 1000) {
            this.handleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error occurred:', error);
          this.isConnecting = false;
          this.emit('connection_error', { error: 'WebSocket connection failed' });

          setTimeout(() => {
            if (this.ws?.readyState !== WebSocket.OPEN) {
              reject(new Error('Backend not available - running in offline mode'));
            }
          }, 2000);
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

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.eventHandlers.clear();
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('connection_failed', { attempts: this.reconnectAttempts });
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);

    setTimeout(() => {
      this.connect(this.url).catch(() => {
        // Reconnection failed, will be handled by the next attempt
      });
    }, delay);
  }

  private handleMessage(message: WebSocketMessage): void {
    // Emit the specific event type
    this.emit(message.type, message.data);
    
    // Also emit a generic 'message' event
    this.emit('message', message);
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

  sendMessage(type: string, data?: any): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const message: WebSocketMessage = { type, data };
      this.ws.send(JSON.stringify(message));
      return true;
    }
    console.warn(`Cannot send message of type ${type}: WebSocket not connected`);
    return false;
  }

  // Chat-specific methods
  sendChatMessage(message: string, agentType?: string): boolean {
    return this.sendMessage('chat_message', {
      message,
      agentType,
      timestamp: new Date().toISOString()
    });
  }

  // Agent-specific methods
  selectAgent(agentType: string): boolean {
    return this.sendMessage('select_agent', { agentType });
  }

  // System methods
  requestSystemStatus(): boolean {
    return this.sendMessage('get_system_status');
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN || false;
  }

  getConnectionState(): string {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'connected';
      case WebSocket.CLOSING: return 'closing';
      case WebSocket.CLOSED: return 'disconnected';
      default: return 'unknown';
    }
  }
}

// Export singleton instance
export const websocketService = new WebSocketService();

// Export class for testing
export { WebSocketService };

export default websocketService;
