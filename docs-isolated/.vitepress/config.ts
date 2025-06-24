import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'PyGent Factory',
  description: 'MCP-Compliant Agent Factory System - Complete Documentation',
  
  // Theme configuration
  themeConfig: {
    // Logo and branding
    logo: '/logo.svg',
    siteTitle: 'PyGent Factory',
    
    // Navigation
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Getting Started', link: '/getting-started/introduction' },
      { text: 'Guides', link: '/guides/' },
      { text: 'API Reference', link: '/api/' },
      { text: 'Examples', link: '/examples/' },
      {
        text: 'v1.0.0',
        items: [
          { text: 'Changelog', link: '/changelog' },
          { text: 'Contributing', link: '/advanced/contributing' }
        ]
      }
    ],

    // Sidebar configuration
    sidebar: {
      '/getting-started/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction', link: '/getting-started/introduction' },
            { text: 'Quick Start', link: '/getting-started/quick-start' },
            { text: 'Installation', link: '/getting-started/installation' },
            { text: 'Your First Agent', link: '/getting-started/first-agent' },
            { text: 'Troubleshooting', link: '/getting-started/troubleshooting' }
          ]
        }
      ],
      
      '/concepts/': [
        {
          text: 'Core Concepts',
          items: [
            { text: 'System Architecture', link: '/concepts/architecture' },
            { text: 'Agent Types', link: '/concepts/agents' },
            { text: 'MCP Protocol', link: '/concepts/mcp-protocol' },
            { text: 'Memory System', link: '/concepts/memory-system' },
            { text: 'RAG System', link: '/concepts/rag-system' },
            { text: 'Communication', link: '/concepts/communication' }
          ]
        }
      ],
      
      '/guides/': [
        {
          text: 'User Guides',
          items: [
            { text: 'Agent Creation', link: '/guides/agent-creation/' },
            { text: 'MCP Servers', link: '/guides/mcp-servers/' },
            { text: 'RAG Setup', link: '/guides/rag-setup/' },
            { text: 'Deployment', link: '/guides/deployment/' },
            { text: 'Monitoring', link: '/guides/monitoring/' },
            { text: 'Best Practices', link: '/guides/best-practices/' }
          ]
        }
      ],
      
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'REST Endpoints', link: '/api/rest-endpoints' },
            { text: 'WebSocket Events', link: '/api/websocket-events' },
            { text: 'Agent Interfaces', link: '/api/agent-interfaces' },
            { text: 'MCP Integration', link: '/api/mcp-integration' },
            { text: 'Python SDK', link: '/api/sdk-reference' }
          ]
        }
      ],
      
      '/examples/': [
        {
          text: 'Examples & Tutorials',
          items: [
            { text: 'Basic Chatbot', link: '/examples/basic-chatbot/' },
            { text: 'Research Agent', link: '/examples/research-agent/' },
            { text: 'Coding Assistant', link: '/examples/coding-assistant/' },
            { text: 'Multi-Agent System', link: '/examples/multi-agent-system/' },
            { text: 'Custom MCP Server', link: '/examples/custom-mcp-server/' }
          ]
        }
      ],
      
      '/advanced/': [
        {
          text: 'Advanced Topics',
          items: [
            { text: 'Custom Agents', link: '/advanced/custom-agents' },
            { text: 'Performance Tuning', link: '/advanced/performance-tuning' },
            { text: 'Security', link: '/advanced/security' },
            { text: 'Scaling', link: '/advanced/scaling' },
            { text: 'Contributing', link: '/advanced/contributing' }
          ]
        }
      ]
    },

    // Social links
    socialLinks: [
      { icon: 'github', link: 'https://github.com/gigamonkeyx/pygent' }
    ],

    // Footer
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 PyGent Factory'
    },

    // Search
    search: {
      provider: 'local'
    },

    // Edit link
    editLink: {
      pattern: 'https://github.com/gigamonkeyx/pygent/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    // Last updated
    lastUpdated: {
      text: 'Updated at',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    }
  },

  // Markdown configuration
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true
  },

  // Mermaid support
  mermaid: {
    // Mermaid configuration
  },

  // Head configuration
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#2563eb' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:locale', content: 'en' }],
    ['meta', { property: 'og:title', content: 'PyGent Factory | MCP-Compliant Agent Factory' }],
    ['meta', { property: 'og:site_name', content: 'PyGent Factory' }],
    ['meta', { property: 'og:image', content: '/og-image.png' }],
    ['meta', { property: 'og:url', content: 'https://docs.pygent.ai/' }]
  ],

  // Build configuration for hybrid architecture
  outDir: './dist',
  cacheDir: './.vitepress/cache',

  // Development server
  server: {
    port: 3001,
    host: true
  },

  // Base URL - remove to fix path resolution
  // base: '/docs/',

  // Vite configuration - completely disable PostCSS to avoid TailwindCSS conflicts
  vite: {
    configFile: './vite.config.ts', // Use our specific vite config
    server: {
      fs: {
        allow: ['..', '../..'] // Allow access to parent directories for CSS files
      }
    },
    css: {
      // Completely disable PostCSS to avoid TailwindCSS conflicts
      postcss: false,
      preprocessorOptions: {
        // Disable all CSS preprocessing
      }
    },
    define: {
      'process.env.NODE_ENV': '"development"'
    },
    // Ensure complete isolation from parent project
    resolve: {
      alias: {
        // Prevent any parent directory imports that might include TailwindCSS
        '@': false,
        '~': false
      }
    },
    // Explicitly exclude TailwindCSS from processing
    optimizeDeps: {
      exclude: ['tailwindcss', '@tailwindcss/postcss']
    }
  }
})