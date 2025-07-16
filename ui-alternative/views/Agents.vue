```vue
<script setup>
import { ref } from 'vue';
import axios from 'axios';

const route = useRoute();
const agentsList = ref([]);
const loading = ref(true);
const error = ref(null);

// Agent management functions would be implemented here

async function fetchAgents() {
  try {
    const response = await axios.get('/api/agents');
    agentsList.value = response.data;
    loading.value = false;
  } catch (err) {
    console.error('Error fetching agents:', err);
    error.value = 'Failed to load agents. Please try again later.';
    loading.value = false;
  }
}

onMounted(() => {
  fetchAgents();
});
</script>

<template>
  <div class="agents-view">
    <h1>Agent Management</h1>
    
    <!-- Agent List -->
    <AgentList :agents="agentsList" v-if="!loading && !error" />
    
    <!-- Error message -->
    <p v-if="error">{{ error }}</p>
    
    <!-- Loading indicator -->
    <div class="loader" v-else-if="loading"></div>
  </div>
</template>

<style scoped>
.agents-view {
  padding: 20px;
}

.agent-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}
```
