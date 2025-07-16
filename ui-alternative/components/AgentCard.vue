```vue
<script setup>
import { ref } from 'vue';
import axios from 'axios';

const props = defineProps({
  agent: {
    type: Object,
    required: true
  }
});

// Agent details would be fetched here in real implementation
</script>

<template>
  <div class="agent-card">
    <h3>{{ agent.name }}</h3>
    <p>Status: {{ agent.status }}</p>
    <p>Model: {{ agent.model }}</p>
    <p>Last Updated: {{ formatDate(agent.updatedAt) }}</p>
    
    <!-- Action buttons -->
    <button @click="handleAction('start')">Start</button>
    <button @click="handleAction('stop')">Stop</button>
  </div>
</template>

<style scoped>
.agent-card {
  background-color: white;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Card specific styles */
.status-indicator {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 8px;
}
```
