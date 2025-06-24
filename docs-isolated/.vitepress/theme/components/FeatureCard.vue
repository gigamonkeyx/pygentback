<template>
  <div class="feature-card" :class="{ clickable: link }">
    <div class="feature-icon" v-if="icon">
      <component :is="iconComponent" />
    </div>
    
    <h3>{{ title }}</h3>
    <p>{{ description }}</p>
    
    <div v-if="badges" class="feature-badges">
      <span
        v-for="badge in badges"
        :key="badge.text"
        :class="['badge', badge.type]"
      >
        {{ badge.text }}
      </span>
    </div>
    
    <div v-if="link" class="feature-link">
      <a :href="link" class="learn-more">
        Learn more →
      </a>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Badge {
  text: string
  type: 'new' | 'beta' | 'deprecated' | 'stable'
}

interface Props {
  title: string
  description: string
  icon?: string
  link?: string
  badges?: Badge[]
}

const props = defineProps<Props>()

const iconComponent = computed(() => {
  // Simple icon mapping - in a real implementation, you'd use a proper icon library
  const iconMap: Record<string, string> = {
    'agent': '🤖',
    'mcp': '🔌',
    'rag': '🧠',
    'websocket': '⚡',
    'memory': '💾',
    'search': '🔍',
    'evolution': '🧬',
    'reasoning': '🤔',
    'api': '🔗',
    'deployment': '🚀'
  }
  
  return iconMap[props.icon || ''] || '📄'
})
</script>

<style scoped>
.feature-card {
  position: relative;
  transition: all 0.3s ease;
}

.feature-card.clickable {
  cursor: pointer;
}

.feature-card.clickable:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}

.feature-icon {
  font-size: 2rem;
  margin-bottom: var(--space-md);
  display: flex;
  align-items: center;
  justify-content: flex-start;
}

.feature-card h3 {
  font-family: 'Inter', var(--vp-font-family-base);
  font-weight: 600;
  font-size: 1.25rem;
  line-height: 1.3;
}

.feature-card p {
  line-height: 1.6;
  margin-bottom: var(--space-md);
}

.feature-badges {
  display: flex;
  gap: var(--space-xs);
  margin-bottom: var(--space-md);
  flex-wrap: wrap;
}

.feature-link {
  margin-top: auto;
}

.learn-more {
  color: var(--vp-c-brand-1);
  font-weight: 600;
  text-decoration: none;
  font-size: 14px;
  display: inline-flex;
  align-items: center;
  gap: var(--space-xs);
  transition: color 0.2s ease;
}

.learn-more:hover {
  color: var(--vp-c-brand-2);
}

/* Grid layout when used in groups */
.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: var(--space-lg);
  margin: var(--space-lg) 0;
}

@media (max-width: 768px) {
  .feature-grid {
    grid-template-columns: 1fr;
  }
}
</style>