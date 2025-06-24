import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import './custom.css'

// Custom components
import CodeExample from './components/CodeExample.vue'
import ArchitectureDiagram from './components/ArchitectureDiagram.vue'
import InteractiveDemo from './components/InteractiveDemo.vue'
import FeatureCard from './components/FeatureCard.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // Custom layout slots can be added here
    })
  },
  enhanceApp({ app, router, siteData }) {
    // Register global components
    app.component('CodeExample', CodeExample)
    app.component('ArchitectureDiagram', ArchitectureDiagram)
    app.component('InteractiveDemo', InteractiveDemo)
    app.component('FeatureCard', FeatureCard)
  }
}