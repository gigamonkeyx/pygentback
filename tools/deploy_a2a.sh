#!/bin/bash

# A2A Multi-Agent System Deployment Script

set -e

echo "🚀 DEPLOYING A2A MULTI-AGENT SYSTEM"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_status "Docker and Docker Compose are available"

# Create logs directory
mkdir -p logs
print_status "Created logs directory"

# Stop any existing containers
print_info "Stopping existing containers..."
docker-compose -f docker-compose.a2a.yml down --remove-orphans || true

# Build and start services
print_info "Building and starting A2A services..."
docker-compose -f docker-compose.a2a.yml up --build -d

# Wait for services to be healthy
print_info "Waiting for services to be healthy..."

# Wait for PostgreSQL
print_info "Waiting for PostgreSQL..."
timeout=60
counter=0
while ! docker-compose -f docker-compose.a2a.yml exec -T postgres pg_isready -U postgres > /dev/null 2>&1; do
    if [ $counter -ge $timeout ]; then
        print_error "PostgreSQL failed to start within $timeout seconds"
        exit 1
    fi
    sleep 1
    counter=$((counter + 1))
done
print_status "PostgreSQL is ready"

# Wait for Redis
print_info "Waiting for Redis..."
counter=0
while ! docker-compose -f docker-compose.a2a.yml exec -T redis redis-cli ping > /dev/null 2>&1; do
    if [ $counter -ge $timeout ]; then
        print_error "Redis failed to start within $timeout seconds"
        exit 1
    fi
    sleep 1
    counter=$((counter + 1))
done
print_status "Redis is ready"

# Wait for A2A server
print_info "Waiting for A2A server..."
counter=0
while ! curl -f http://localhost:8080/health > /dev/null 2>&1; do
    if [ $counter -ge $timeout ]; then
        print_error "A2A server failed to start within $timeout seconds"
        print_info "Checking A2A server logs..."
        docker-compose -f docker-compose.a2a.yml logs a2a_server
        exit 1
    fi
    sleep 2
    counter=$((counter + 2))
done
print_status "A2A server is ready"

# Display service status
echo ""
print_status "A2A MULTI-AGENT SYSTEM DEPLOYED SUCCESSFULLY!"
echo "=============================================="

print_info "Service Status:"
docker-compose -f docker-compose.a2a.yml ps

echo ""
print_info "Available Endpoints:"
echo "  🔗 A2A JSON-RPC API: http://localhost:8080/"
echo "  📡 Agent Discovery: http://localhost:8080/.well-known/agent.json"
echo "  🏥 Health Check: http://localhost:8080/health"
echo "  👥 Agents List: http://localhost:8080/agents"
echo "  🗄️ PostgreSQL: localhost:54321"
echo "  🔴 Redis: localhost:6379"

echo ""
print_info "Testing the deployment..."

# Test agent discovery
if curl -s http://localhost:8080/.well-known/agent.json | jq . > /dev/null 2>&1; then
    print_status "Agent discovery endpoint working"
else
    print_warning "Agent discovery endpoint may not be working properly"
fi

# Test health check
health_response=$(curl -s http://localhost:8080/health)
if echo "$health_response" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
    agents_count=$(echo "$health_response" | jq -r '.agents_registered')
    print_status "Health check passed - $agents_count agents registered"
else
    print_warning "Health check may not be working properly"
fi

echo ""
print_info "Deployment Commands:"
echo "  📊 View logs: docker-compose -f docker-compose.a2a.yml logs -f"
echo "  🛑 Stop system: docker-compose -f docker-compose.a2a.yml down"
echo "  🔄 Restart: docker-compose -f docker-compose.a2a.yml restart"
echo "  🧪 Run demo: docker-compose -f docker-compose.a2a.yml --profile demo up a2a_demo"

echo ""
print_status "A2A Multi-Agent System is now running!"
print_info "You can now run the live demonstration with: python a2a_live_demo.py"
