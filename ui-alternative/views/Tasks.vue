```vue
<script setup>
import { ref } from 'vue';
import axios from 'axios';

const route = useRoute();
const tasksList = ref([]);
const loading = ref(true);
const error = ref(null);

// Task management functions would be implemented here

async function fetchTasks() {
  try {
    const response = await axios.get('/api/tasks');
    tasksList.value = response.data;
    loading.value = false;
  } catch (err) {
    console.error('Error fetching tasks:', err);
    error.value = 'Failed to load tasks. Please try again later.';
    loading.value = false;
  }
}

onMounted(() => {
  fetchTasks();
});
</script>

<template>
  <div class="tasks-view">
    <h1>Task Management</h1>
    
    <!-- Task List -->
    <TaskList :tasks="tasksList" v-if="!loading && !error" />
    
    <!-- Error message -->
    <p v-if="error">{{ error }}</p>
    
    <!-- Loading indicator -->
    <div class="loader" v-else-if="loading"></div>
  </div>
</template>

<style scoped>
.tasks-view {
  padding: 20px;
}

.task-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 20px;
}
```
