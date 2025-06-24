<template>
  <div class="interactive-demo">
    <h4>{{ title }}</h4>
    <p v-if="description">{{ description }}</p>
    
    <div class="demo-content">
      <div v-if="type === 'api'" class="api-demo">
        <div class="demo-input">
          <label>API Endpoint:</label>
          <input v-model="apiEndpoint" type="text" placeholder="/api/v1/agents" />
        </div>
        <div class="demo-input">
          <label>Request Body:</label>
          <textarea v-model="requestBody" placeholder='{"message": "Hello, agent!"}'></textarea>
        </div>
        <button @click="makeApiCall" :disabled="loading" class="demo-button">
          {{ loading ? 'Sending...' : 'Send Request' }}
        </button>
        <div v-if="response" class="demo-response">
          <h5>Response:</h5>
          <pre><code>{{ response }}</code></pre>
        </div>
      </div>
      
      <div v-else-if="type === 'websocket'" class="websocket-demo">
        <div class="connection-status" :class="{ connected: wsConnected }">
          {{ wsConnected ? '🟢 Connected' : '🔴 Disconnected' }}
        </div>
        <button @click="toggleWebSocket" class="demo-button">
          {{ wsConnected ? 'Disconnect' : 'Connect to WebSocket' }}
        </button>
        <div v-if="wsConnected" class="demo-input">
          <input v-model="wsMessage" @keyup.enter="sendWebSocketMessage" placeholder="Type a message..." />
          <button @click="sendWebSocketMessage" class="demo-button">Send</button>
        </div>
        <div v-if="wsMessages.length" class="demo-messages">
          <h5>Messages:</h5>
          <div v-for="(msg, index) in wsMessages" :key="index" class="message">
            <span class="timestamp">{{ msg.timestamp }}</span>
            <span class="content">{{ msg.content }}</span>
          </div>
        </div>
      </div>
      
      <div v-else class="placeholder-demo">
        <p>Interactive demo will be available here</p>
        <button class="demo-button" disabled>Coming Soon</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue'

interface Props {
  title: string
  description?: string
  type: 'api' | 'websocket' | 'placeholder'
  endpoint?: string
}

const props = defineProps<Props>()

// API Demo state
const apiEndpoint = ref(props.endpoint || '/api/v1/agents')
const requestBody = ref('{"message": "Hello, agent!"}')
const response = ref('')
const loading = ref(false)

// WebSocket Demo state
const wsConnected = ref(false)
const wsMessage = ref('')
const wsMessages = ref<Array<{ timestamp: string; content: string }>>([])
let websocket: WebSocket | null = null

const makeApiCall = async () => {
  loading.value = true
  response.value = ''
  
  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    response.value = JSON.stringify({
      status: 'success',
      data: {
        agent_id: 'demo-agent-123',
        response: 'Hello! I\'m a PyGent Factory agent.',
        timestamp: new Date().toISOString()
      }
    }, null, 2)
  } catch (error) {
    response.value = JSON.stringify({
      status: 'error',
      message: 'Demo API call failed'
    }, null, 2)
  } finally {
    loading.value = false
  }
}

const toggleWebSocket = () => {
  if (wsConnected.value) {
    disconnectWebSocket()
  } else {
    connectWebSocket()
  }
}

const connectWebSocket = () => {
  try {
    // Simulate WebSocket connection
    wsConnected.value = true
    addMessage('Connected to PyGent Factory WebSocket')
    
    // Simulate receiving messages
    setTimeout(() => {
      if (wsConnected.value) {
        addMessage('Welcome! You can now send messages to agents.')
      }
    }, 1000)
  } catch (error) {
    addMessage('Failed to connect to WebSocket')
  }
}

const disconnectWebSocket = () => {
  if (websocket) {
    websocket.close()
    websocket = null
  }
  wsConnected.value = false
  addMessage('Disconnected from WebSocket')
}

const sendWebSocketMessage = () => {
  if (!wsMessage.value.trim()) return
  
  addMessage(`You: ${wsMessage.value}`)
  
  // Simulate agent response
  setTimeout(() => {
    addMessage(`Agent: I received your message: "${wsMessage.value}"`)
  }, 500)
  
  wsMessage.value = ''
}

const addMessage = (content: string) => {
  wsMessages.value.push({
    timestamp: new Date().toLocaleTimeString(),
    content
  })
  
  // Keep only last 10 messages
  if (wsMessages.value.length > 10) {
    wsMessages.value = wsMessages.value.slice(-10)
  }
}

onUnmounted(() => {
  disconnectWebSocket()
})
</script>

<style scoped>
.interactive-demo h4 {
  margin-bottom: var(--space-sm);
}

.demo-content {
  margin-top: var(--space-lg);
}

.demo-input {
  margin-bottom: var(--space-md);
}

.demo-input label {
  display: block;
  margin-bottom: var(--space-xs);
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.demo-input input,
.demo-input textarea {
  width: 100%;
  padding: var(--space-sm);
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--radius-md);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
}

.demo-input textarea {
  min-height: 80px;
  resize: vertical;
}

.demo-button {
  background: var(--vp-c-brand-1);
  color: white;
  border: none;
  padding: var(--space-sm) var(--space-md);
  border-radius: var(--radius-md);
  cursor: pointer;
  font-weight: 600;
  transition: background-color 0.2s ease;
}

.demo-button:hover:not(:disabled) {
  background: var(--vp-c-brand-2);
}

.demo-button:disabled {
  background: var(--vp-c-divider);
  cursor: not-allowed;
}

.demo-response {
  margin-top: var(--space-lg);
  padding: var(--space-md);
  background: var(--vp-c-bg-soft);
  border-radius: var(--radius-md);
  border: 1px solid var(--vp-c-divider);
}

.demo-response h5 {
  margin: 0 0 var(--space-sm) 0;
  color: var(--vp-c-text-1);
}

.demo-response pre {
  margin: 0;
  background: var(--vp-code-block-bg);
  padding: var(--space-sm);
  border-radius: var(--radius-sm);
  overflow-x: auto;
}

.connection-status {
  padding: var(--space-sm);
  border-radius: var(--radius-md);
  margin-bottom: var(--space-md);
  font-weight: 600;
  background: var(--vp-c-red-soft);
  color: var(--vp-c-red-1);
}

.connection-status.connected {
  background: var(--vp-c-green-soft);
  color: var(--vp-c-green-1);
}

.demo-messages {
  margin-top: var(--space-lg);
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--radius-md);
  padding: var(--space-md);
}

.demo-messages h5 {
  margin: 0 0 var(--space-sm) 0;
  color: var(--vp-c-text-1);
}

.message {
  display: flex;
  gap: var(--space-sm);
  margin-bottom: var(--space-xs);
  font-size: 14px;
}

.timestamp {
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  flex-shrink: 0;
}

.content {
  color: var(--vp-c-text-1);
}

.placeholder-demo {
  text-align: center;
  padding: var(--space-xl);
  color: var(--vp-c-text-2);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .demo-input input,
  .demo-input textarea {
    font-size: 16px; /* Prevent zoom on iOS */
  }
  
  .message {
    flex-direction: column;
    gap: var(--space-xs);
  }
}
</style>