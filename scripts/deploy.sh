#!/bin/bash

# PyGent Factory Deployment Script
# This script handles the complete deployment of PyGent Factory with UI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Node.js (for UI build)
    if ! command -v node &> /dev/null; then
        log_warning "Node.js is not installed. UI will be built in Docker container."
    fi
    
    # Check available disk space (minimum 5GB)
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 5242880 ]; then  # 5GB in KB
        log_warning "Less than 5GB disk space available. Deployment may fail."
    fi
    
    log_success "System requirements check completed"
}

setup_environment() {
    log_info "Setting up environment configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating .env file from template..."
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        
        # Generate secure secrets
        SECRET_KEY=$(openssl rand -hex 32)
        JWT_SECRET=$(openssl rand -hex 32)
        POSTGRES_PASSWORD=$(openssl rand -hex 16)
        REDIS_PASSWORD=$(openssl rand -hex 16)
        
        # Update .env file with generated secrets
        sed -i "s/your_super_secret_key_change_in_production_minimum_32_characters/$SECRET_KEY/g" "$ENV_FILE"
        sed -i "s/your_jwt_secret_change_in_production_minimum_32_characters/$JWT_SECRET/g" "$ENV_FILE"
        sed -i "s/pygent_secure_password_change_me/$POSTGRES_PASSWORD/g" "$ENV_FILE"
        sed -i "s/redis_secure_password_change_me/$REDIS_PASSWORD/g" "$ENV_FILE"
        
        log_success "Environment file created with secure secrets"
        log_warning "Please review and update .env file with your API keys and configuration"
    else
        log_info "Environment file already exists"
    fi
}

build_frontend() {
    log_info "Building frontend application..."
    
    cd "$PROJECT_ROOT/ui"
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm install
    fi
    
    # Build the frontend
    log_info "Building production frontend..."
    npm run build
    
    log_success "Frontend build completed"
    cd "$PROJECT_ROOT"
}

start_services() {
    log_info "Starting PyGent Factory services..."
    
    # Pull latest images
    log_info "Pulling Docker images..."
    docker-compose pull
    
    # Build custom images
    log_info "Building custom Docker images..."
    docker-compose build
    
    # Start services
    log_info "Starting all services..."
    docker-compose up -d
    
    log_success "Services started successfully"
}

wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for database
    log_info "Waiting for PostgreSQL..."
    timeout=60
    while ! docker-compose exec -T postgres pg_isready -U postgres -d pygent_factory &> /dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            log_error "PostgreSQL failed to start within 60 seconds"
            exit 1
        fi
    done
    log_success "PostgreSQL is ready"
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    timeout=30
    while ! docker-compose exec -T redis redis-cli ping &> /dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            log_error "Redis failed to start within 30 seconds"
            exit 1
        fi
    done
    log_success "Redis is ready"
    
    # Wait for ChromaDB
    log_info "Waiting for ChromaDB..."
    timeout=60
    while ! curl -f http://localhost:8001/api/v1/heartbeat &> /dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            log_error "ChromaDB failed to start within 60 seconds"
            exit 1
        fi
    done
    log_success "ChromaDB is ready"
    
    # Wait for backend API
    log_info "Waiting for PyGent Factory API..."
    timeout=120
    while ! curl -f http://localhost:8080/api/v1/health &> /dev/null; do
        sleep 5
        timeout=$((timeout - 5))
        if [ $timeout -le 0 ]; then
            log_error "PyGent Factory API failed to start within 120 seconds"
            exit 1
        fi
    done
    log_success "PyGent Factory API is ready"
    
    # Wait for frontend
    log_info "Waiting for Frontend..."
    timeout=60
    while ! curl -f http://localhost:3000/health &> /dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            log_error "Frontend failed to start within 60 seconds"
            exit 1
        fi
    done
    log_success "Frontend is ready"
}

show_status() {
    log_info "Deployment Status:"
    echo ""
    echo "🌐 Frontend UI:      http://localhost:3000"
    echo "🔧 Backend API:      http://localhost:8080"
    echo "📊 API Docs:         http://localhost:8080/docs"
    echo "🗄️  PostgreSQL:      localhost:54321"
    echo "🔴 Redis:            localhost:6379"
    echo "🔍 ChromaDB:         http://localhost:8001"
    echo ""
    echo "📋 Service Status:"
    docker-compose ps
    echo ""
    log_success "PyGent Factory is now running!"
    echo ""
    echo "To stop all services: docker-compose down"
    echo "To view logs: docker-compose logs -f"
    echo "To restart a service: docker-compose restart <service_name>"
}

cleanup_on_error() {
    log_error "Deployment failed. Cleaning up..."
    docker-compose down
    exit 1
}

# Main deployment process
main() {
    log_info "Starting PyGent Factory deployment..."
    echo ""
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Run deployment steps
    check_requirements
    setup_environment
    
    # Ask user if they want to build frontend locally or in Docker
    if command -v node &> /dev/null; then
        read -p "Build frontend locally? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            build_frontend
        fi
    fi
    
    start_services
    wait_for_services
    show_status
    
    log_success "Deployment completed successfully!"
}

# Parse command line arguments
case "${1:-}" in
    "stop")
        log_info "Stopping PyGent Factory services..."
        docker-compose down
        log_success "Services stopped"
        ;;
    "restart")
        log_info "Restarting PyGent Factory services..."
        docker-compose restart
        log_success "Services restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        log_warning "This will remove all containers, volumes, and images. Are you sure? (y/N)"
        read -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v --rmi all
            log_success "Cleanup completed"
        fi
        ;;
    *)
        main
        ;;
esac
