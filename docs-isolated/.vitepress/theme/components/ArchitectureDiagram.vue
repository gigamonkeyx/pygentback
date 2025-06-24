<template>
  <div class="architecture-diagram">
    <h4 v-if="title">{{ title }}</h4>
    <div class="diagram-container" ref="diagramContainer">
      <div v-if="type === 'mermaid'" class="mermaid">
        {{ content }}
      </div>
      <div v-else-if="type === 'svg'" v-html="content" />
      <div v-else class="diagram-placeholder">
        <p>{{ content || 'Diagram will be rendered here' }}</p>
      </div>
    </div>
    <div v-if="description" class="diagram-description">
      <p>{{ description }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'

interface Props {
  title?: string
  type: 'mermaid' | 'svg' | 'placeholder'
  content: string
  description?: string
}

const props = defineProps<Props>()
const diagramContainer = ref<HTMLElement>()

onMounted(async () => {
  if (props.type === 'mermaid') {
    await nextTick()
    // Mermaid will be initialized by the markdown-it plugin
    // This is just the container
  }
})
</script>

<style scoped>
.architecture-diagram h4 {
  margin-top: 0;
  margin-bottom: var(--space-md);
  color: var(--vp-c-brand-1);
  font-family: 'Inter', var(--vp-font-family-base);
}

.diagram-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 200px;
}

.diagram-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--vp-c-bg-soft);
  border: 2px dashed var(--vp-c-divider);
  border-radius: var(--radius-md);
  padding: var(--space-xl);
  width: 100%;
  min-height: 200px;
}

.diagram-placeholder p {
  color: var(--vp-c-text-2);
  font-style: italic;
  margin: 0;
}

.diagram-description {
  margin-top: var(--space-md);
  padding: var(--space-md);
  background: var(--vp-c-bg-soft);
  border-radius: var(--radius-md);
  border-left: 4px solid var(--vp-c-brand-1);
}

.diagram-description p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 14px;
}

/* Mermaid diagram styling */
:deep(.mermaid) {
  display: flex;
  justify-content: center;
}

:deep(.mermaid svg) {
  max-width: 100%;
  height: auto;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .diagram-container {
    overflow-x: auto;
  }
  
  :deep(.mermaid svg) {
    min-width: 600px;
  }
}
</style>