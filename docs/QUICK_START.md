# PyGent Factory - Quick Start Guide

Get PyGent Factory with UI up and running in minutes!

## 🚀 **One-Command Deployment**

```bash
# Clone and deploy in one go
git clone https://github.com/your-org/pygent-factory.git
cd pygent-factory
./scripts/deploy.sh
```

## 📋 **Prerequisites**

- **Docker** and **Docker Compose** installed
- **8GB+ RAM** recommended
- **5GB+ free disk space**
- **NVIDIA GPU** (optional, for GPU acceleration)

## 🎯 **Quick Deploy Options**

### Option 1: Full Stack with Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Option 2: Development Mode
```bash
# Backend only
python -m src.api.main

# Frontend (separate terminal)
cd ui
npm install
npm run dev
```

### Option 3: Production Deployment
```bash
# Use production profile
docker-compose --profile production up -d
```

## 🌐 **Access Points**

Once deployed, access these URLs:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend UI** | http://localhost:3000 | Main user interface |
| **Backend API** | http://localhost:8080 | REST API endpoints |
| **API Documentation** | http://localhost:8080/docs | Interactive API docs |
| **WebSocket** | ws://localhost:8080/ws | Real-time communication |
| **PostgreSQL** | localhost:54321 | Database (user: postgres) |
| **Redis** | localhost:6379 | Cache and message queue |
| **ChromaDB** | http://localhost:8001 | Vector database |

## 🔧 **Configuration**

### Environment Setup
```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

### Key Configuration Options
```env
# API Keys (required for full functionality)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# GPU Support
PYGENT_GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0

# Feature Flags
FEATURE_REASONING_ENABLED=true
FEATURE_EVOLUTION_ENABLED=true
FEATURE_SEARCH_ENABLED=true
FEATURE_MCP_ENABLED=true
```

## 🎮 **Using the UI**

### 1. **Login**
- Navigate to http://localhost:3000
- Use any username/password (demo mode)
- Click "Sign In"

### 2. **Chat with AI Agents**
- Select agent type (Reasoning, Evolution, Search, General)
- Type your message and press Enter
- Watch real-time responses and reasoning

### 3. **Explore Features**
- **Reasoning Panel**: View thought trees and reasoning paths
- **System Monitoring**: Check CPU, GPU, memory usage
- **MCP Servers**: Manage Model Context Protocol servers
- **Settings**: Configure preferences and themes

## 🔍 **Testing the System**

### Quick Health Check
```bash
# Check all services
curl http://localhost:8080/api/v1/health
curl http://localhost:3000/health

# Test WebSocket connection
wscat -c ws://localhost:8080/ws
```

### Sample API Requests
```bash
# Get system status
curl http://localhost:8080/api/v1/health

# Test reasoning endpoint
curl -X POST http://localhost:8080/api/v1/agents/reasoning \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the meaning of life?"}'

# List MCP servers
curl http://localhost:8080/api/v1/mcp/servers
```

## 🛠️ **Troubleshooting**

### Common Issues

**Port Conflicts**
```bash
# Check what's using ports
netstat -tulpn | grep :3000
netstat -tulpn | grep :8080

# Change ports in docker-compose.yml if needed
```

**Service Won't Start**
```bash
# Check logs
docker-compose logs <service_name>

# Restart specific service
docker-compose restart <service_name>

# Full restart
docker-compose down && docker-compose up -d
```

**Frontend Build Issues**
```bash
# Clear npm cache
cd ui
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**Database Connection Issues**
```bash
# Reset database
docker-compose down -v
docker-compose up -d postgres
# Wait for postgres to be ready, then start other services
```

### Performance Optimization

**GPU Acceleration**
```bash
# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Enable GPU in environment
echo "PYGENT_GPU_ENABLED=true" >> .env
```

**Memory Optimization**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory > 8GB+

# Monitor memory usage
docker stats
```

## 📊 **Monitoring**

### System Metrics
- **Frontend**: Real-time dashboard at http://localhost:3000
- **API Metrics**: Available at http://localhost:8080/metrics
- **Container Stats**: `docker stats`

### Log Management
```bash
# View all logs
docker-compose logs -f

# Service-specific logs
docker-compose logs -f frontend
docker-compose logs -f backend
docker-compose logs -f postgres

# Log rotation (production)
docker-compose logs --tail=100 -f
```

## 🔄 **Updates and Maintenance**

### Updating the System
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup and Restore
```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres pygent_factory > backup.sql

# Backup volumes
docker run --rm -v pygent-factory_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data

# Restore database
docker-compose exec -T postgres psql -U postgres pygent_factory < backup.sql
```

## 🚨 **Production Considerations**

### Security
- Change default passwords in `.env`
- Use HTTPS in production
- Configure firewall rules
- Enable authentication
- Regular security updates

### Scaling
- Use load balancer for multiple frontend instances
- Scale backend with multiple workers
- Use external database for production
- Implement proper monitoring and alerting

### Performance
- Enable Redis caching
- Configure CDN for static assets
- Optimize database queries
- Monitor and tune GPU usage

## 📞 **Support**

### Getting Help
- **Documentation**: Check the full README.md
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub Discussions for questions
- **Logs**: Always include relevant logs when reporting issues

### Useful Commands
```bash
# Complete system reset
./scripts/deploy.sh clean

# View system status
./scripts/deploy.sh status

# Stop all services
./scripts/deploy.sh stop

# Restart all services
./scripts/deploy.sh restart
```

---

## 🎉 **You're Ready!**

PyGent Factory is now running with a complete UI system. Start exploring the AI capabilities:

1. **Chat with AI agents** for reasoning and problem-solving
2. **Monitor system performance** in real-time
3. **Manage MCP servers** for extended functionality
4. **Explore evolution algorithms** for optimization
5. **Use vector search** for document retrieval

**Happy AI reasoning! 🤖✨**
