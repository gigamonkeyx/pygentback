<template>
  <div class="code-example">
    <div class="code-example-tabs" v-if="tabs.length > 1">
      <button
        v-for="(tab, index) in tabs"
        :key="tab.name"
        :class="['code-example-tab', { active: activeTab === index }]"
        @click="activeTab = index"
      >
        {{ tab.name }}
      </button>
    </div>
    
    <div class="code-example-content">
      <div v-html="tabs[activeTab]?.content || ''" />
    </div>
    
    <div v-if="description" class="code-example-description">
      <p>{{ description }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface CodeTab {
  name: string
  content: string
  language?: string
}

interface Props {
  tabs?: CodeTab[]
  code?: string
  language?: string
  title?: string
  description?: string
}

const props = withDefaults(defineProps<Props>(), {
  tabs: () => [],
  language: 'python',
  title: '',
  description: ''
})

const activeTab = ref(0)

const tabs = computed(() => {
  if (props.tabs.length > 0) {
    return props.tabs
  }
  
  if (props.code) {
    return [{
      name: props.title || props.language,
      content: `<pre><code class="language-${props.language}">${props.code}</code></pre>`
    }]
  }
  
  return []
})
</script>

<style scoped>
.code-example-content {
  background: var(--vp-code-block-bg);
  border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}

.code-example-content :deep(pre) {
  margin: 0;
  border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}

.code-example-description {
  margin-top: var(--space-md);
  padding: var(--space-md);
  background: var(--vp-c-bg-soft);
  border-radius: var(--radius-md);
  border-left: 4px solid var(--vp-c-brand-1);
}

.code-example-description p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 14px;
}
</style>