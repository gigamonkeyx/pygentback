```vue
<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';

const route = useRoute();
const stats = ref({ agents: 0, tasks: 0, results: 0 });
const recentTasks = ref([]);
const notifications = ref([]);

onMounted(async () => {
  try {
    const response = await axios.get('/api/dashboard/stats');
    stats.value = response.data;

    const taskResponse = await axios.get('/api/tasks/recent');
    recentTasks.value = taskResponse.data;

    // Mock notification data (would come from API in real app)
    notifications.value = [
      { id: 1, message: 'New agent deployment started', time: '5 min ago' },
      { id: 2, message: 'Task completed successfully', time: '10 min ago' }
    ];
  } catch (error) {
    console.error('Failed to fetch dashboard data:', error);
  }
});
</script>

<template>
  <div class="dashboard-view">
    <h1>Dashboard Overview</h1>
    
    <!-- Stats Cards -->
    <StatsCards :stats="stats" />
    
    <!-- Recent Activity -->
    <RecentActivity :tasks="recentTasks" :notifications="notifications" />
  </div>
</template>

<style scoped>
.dashboard-view {
  padding: 20px;
}

/* Dashboard specific styles */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.activity-feed {
  background-color: white;
  border-radius: 8px;
  padding: 20px;
}
</style>
```
