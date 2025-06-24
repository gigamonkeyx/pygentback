import { defineConfig } from 'vite'

// Vite configuration for VitePress documentation
// This ensures complete isolation from parent project configurations
export default defineConfig({
  css: {
    postcss: false // Completely disable PostCSS
  },
  server: {
    fs: {
      allow: ['..', '../..'] // Allow access to parent directories for assets
    }
  },
  resolve: {
    alias: {
      // Prevent any imports from parent directories that might include TailwindCSS
      '@': false
    }
  }
})
