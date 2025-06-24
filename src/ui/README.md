# PyGent Factory UI

A modern, responsive React-based user interface for the PyGent Factory AI reasoning system.

## 🚀 Features

### 🤖 Multi-Agent Chat Interface
- Real-time conversations with specialized AI agents
- Support for reasoning, evolution, search, and general agents
- Rich message rendering with markdown and code highlighting
- Typing indicators and real-time message streaming

### 🧠 Tree of Thought Reasoning
- Interactive visualization of reasoning paths
- Real-time thought tree updates
- Confidence scoring and path exploration
- Detailed reasoning step analysis

### 📊 System Monitoring
- Real-time system metrics (CPU, Memory, GPU)
- AI agent performance tracking
- Network and health monitoring
- Automated alerting system

### 📦 MCP Marketplace
- Discover and install Model Context Protocol servers
- Manage server lifecycle (install, start, stop)
- Browse server capabilities and documentation
- Integration with PyGent Factory orchestration

### ⚙️ Settings & Configuration
- Theme customization (light, dark, system)
- Notification preferences
- Agent configuration
- System information

## 🛠️ Technology Stack

- **Frontend**: React 18 + TypeScript
- **State Management**: Zustand
- **UI Components**: Radix UI + Tailwind CSS
- **Real-time Communication**: WebSocket + Socket.IO
- **Data Visualization**: Recharts + D3.js
- **Build Tool**: Vite
- **Code Quality**: ESLint + TypeScript

## 🏗️ Architecture

```
src/
├── components/          # React components
│   ├── chat/           # Chat interface components
│   ├── layout/         # Layout components (sidebar, header)
│   └── ui/             # Reusable UI components
├── pages/              # Page components
├── services/           # API and WebSocket services
├── stores/             # Zustand state management
├── types/              # TypeScript type definitions
└── utils/              # Utility functions
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- PyGent Factory backend running on port 8000

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Open browser**:
   Navigate to `http://localhost:3000`

### Backend Integration

The UI connects to the PyGent Factory backend services:

- **API Backend**: `http://localhost:8000` (development) / `https://api.timpayne.net` (production)
- **WebSocket**: `ws://localhost:8000/ws` (development) / `wss://ws.timpayne.net/ws` (production)
- **ToT Agent**: `http://localhost:8001`
- **RAG Agent**: `http://localhost:8002`

## 🔧 Development

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run lint:fix     # Fix ESLint issues
npm run test         # Run tests
```

### Environment Variables

Create a `.env.local` file for local development:

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
VITE_ENABLE_ANALYTICS=false
```

For production deployment:

```bash
VITE_API_BASE_URL=https://api.timpayne.net
VITE_WS_BASE_URL=wss://ws.timpayne.net
VITE_ENABLE_ANALYTICS=true
```

## 📦 Build & Deployment

### Local Build

```bash
npm run build
npm run preview
```

### Production Deployment

The UI is designed for deployment to Cloudflare Pages:

1. **Build the application**:
   ```bash
   npm run build
   ```

2. **Deploy to Cloudflare Pages**:
   - Connect GitHub repository
   - Set build command: `npm run build`
   - Set build output directory: `dist`
   - Configure environment variables

### Performance Optimization

- **Code Splitting**: Automatic route-based splitting
- **Bundle Analysis**: `npm run build:analyze`
- **Lazy Loading**: Components loaded on demand
- **Caching**: Aggressive caching for static assets

## 🔌 WebSocket Integration

The UI maintains real-time connections with the backend:

### Supported Events

- `chat_message` / `chat_response` - Chat interactions
- `reasoning_update` / `reasoning_complete` - ToT reasoning
- `system_metrics` - Real-time system monitoring
- `mcp_server_status` - MCP server health

### Connection Management

- **Automatic Reconnection**: Exponential backoff strategy
- **Connection State Tracking**: Visual indicators
- **Error Handling**: Graceful degradation
- **Message Queuing**: During disconnections

## 🎨 UI/UX Design

### Design System

- **Colors**: CSS custom properties for theming
- **Typography**: Inter font family with size scale
- **Spacing**: Consistent 4px grid system
- **Components**: Radix UI primitives with custom styling

### Responsive Design

- **Mobile First**: Optimized for mobile devices
- **Breakpoints**: Tailwind CSS responsive utilities
- **Touch Friendly**: Large touch targets
- **Progressive Disclosure**: Collapsible sidebar

### Accessibility

- **WCAG Compliance**: Following accessibility guidelines
- **Keyboard Navigation**: Full keyboard support
- **Screen Readers**: Proper ARIA labels
- **Color Contrast**: High contrast ratios

## 🧪 Testing

### Unit Tests

```bash
npm run test
```

### Integration Tests

```bash
npm run test:integration
```

### E2E Tests

```bash
npm run test:e2e
```

## 📈 Performance Monitoring

### Metrics Tracked

- **Bundle Size**: Initial and chunk sizes
- **Load Time**: Time to interactive
- **WebSocket Latency**: Real-time communication
- **Memory Usage**: Client-side memory consumption

### Optimization Targets

- **Initial Bundle**: < 1MB
- **Time to Interactive**: < 3 seconds
- **WebSocket Latency**: < 100ms
- **Memory Usage**: < 50MB

## 🔧 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check backend is running on port 8000
   - Verify WebSocket endpoint is accessible
   - Check firewall/proxy settings

2. **Build Errors**
   - Clear node_modules and reinstall
   - Check Node.js version (18+ required)
   - Verify all dependencies are installed

3. **Performance Issues**
   - Check bundle size with `npm run build:analyze`
   - Monitor memory usage in DevTools
   - Verify WebSocket connection stability

### Debug Mode

Enable debug logging:

```bash
VITE_DEBUG=true npm run dev
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is part of PyGent Factory and follows the same licensing terms.

---

**PyGent Factory UI - Advanced AI Reasoning Interface** 🚀