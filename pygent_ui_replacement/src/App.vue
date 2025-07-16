<template>
  <div id="app" class="min-h-screen bg-gray-100">
    <nav class="bg-blue-600 text-white p-4">
      <h1 class="text-xl font-bold">PyGent Factory UI</h1>
    </nav>
    
    <main class="container mx-auto p-6">
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <!-- Agent Status Cards -->
        <div class="bg-white rounded-lg shadow p-6">
          <h2 class="text-lg font-semibold mb-4">Agent Status</h2>
          <div class="space-y-2">
            <div v-for="agent in agents" :key="agent.id" class="flex justify-between">
              <span>{{ agent.name }}</span>
              <span :class="getStatusColor(agent.status)">{{ agent.status }}</span>
            </div>
          </div>
        </div>
        
        <!-- Task Queue -->
        <div class="bg-white rounded-lg shadow p-6">
          <h2 class="text-lg font-semibold mb-4">Task Queue</h2>
          <div class="space-y-2">
            <div v-for="task in tasks" :key="task.id" class="border-l-4 border-blue-500 pl-3">
              <div class="font-medium">{{ task.name }}</div>
              <div class="text-sm text-gray-600">{{ task.status }}</div>
              <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                <div class="bg-blue-600 h-2 rounded-full" :style="{width: task.progress + '%'}"></div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- System Metrics -->
        <div class="bg-white rounded-lg shadow p-6">
          <h2 class="text-lg font-semibold mb-4">System Metrics</h2>
          <div class="space-y-3">
            <div class="flex justify-between">
              <span>CPU Usage</span>
              <span class="font-mono">{{ metrics.cpu }}%</span>
            </div>
            <div class="flex justify-between">
              <span>Memory</span>
              <span class="font-mono">{{ metrics.memory }}%</span>
            </div>
            <div class="flex justify-between">
              <span>Active Agents</span>
              <span class="font-mono">{{ metrics.activeAgents }}</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

interface Agent {
  id: string
  name: string
  status: 'active' | 'idle' | 'error'
}

interface Task {
  id: string
  name: string
  status: string
  progress: number
}

interface Metrics {
  cpu: number
  memory: number
  activeAgents: number
}

const agents = ref<Agent[]>([
  { id: '1', name: 'Research Agent', status: 'active' },
  { id: '2', name: 'Coding Agent', status: 'idle' },
  { id: '3', name: 'Analysis Agent', status: 'active' }
])

const tasks = ref<Task[]>([
  { id: '1', name: 'UI Component Creation', status: 'In Progress', progress: 75 },
  { id: '2', name: 'API Integration', status: 'Pending', progress: 0 },
  { id: '3', name: 'Testing Suite', status: 'Queued', progress: 0 }
])

const metrics = ref<Metrics>({
  cpu: 45,
  memory: 62,
  activeAgents: 3
})

const getStatusColor = (status: string) => {
  switch (status) {
    case 'active': return 'text-green-600'
    case 'idle': return 'text-yellow-600'
    case 'error': return 'text-red-600'
    default: return 'text-gray-600'
  }
}

onMounted(() => {
  // Simulate real-time updates
  setInterval(() => {
    metrics.value.cpu = Math.floor(Math.random() * 100)
    metrics.value.memory = Math.floor(Math.random() * 100)
  }, 5000)
})
</script>