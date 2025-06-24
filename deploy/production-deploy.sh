#!/bin/bash
# A2A Production Deployment Script
# Deploys PyGent Factory A2A Multi-Agent System to production Kubernetes cluster

set -e  # Exit on any error

# Configuration
NAMESPACE="a2a-system"
DEPLOYMENT_NAME="a2a-server"
SERVICE_NAME="a2a-service"
INGRESS_NAME="a2a-ingress"
TIMEOUT="300s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check if Docker image exists
    if ! docker image inspect pygent-factory/a2a-server:v1.0.0 &> /dev/null; then
        warning "Docker image not found locally, will be pulled from registry"
    fi
    
    success "Prerequisites check completed"
}

# Create namespace
create_namespace() {
    log "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace $NAMESPACE
        success "Namespace $NAMESPACE created"
    fi
    
    # Label namespace
    kubectl label namespace $NAMESPACE environment=production --overwrite
    kubectl label namespace $NAMESPACE app=a2a-system --overwrite
}

# Generate production secrets
generate_secrets() {
    log "Generating production secrets..."
    
    # Generate random passwords if not set
    export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}
    export REDIS_PASSWORD=${REDIS_PASSWORD:-$(openssl rand -base64 32)}
    export A2A_SECRET_KEY=${A2A_SECRET_KEY:-$(openssl rand -base64 64)}
    export A2A_ADMIN_API_KEY=${A2A_ADMIN_API_KEY:-$(openssl rand -hex 32)}
    export A2A_SYSTEM_API_KEY=${A2A_SYSTEM_API_KEY:-$(openssl rand -hex 32)}
    
    # Create secret manifest
    cat > /tmp/a2a-secrets.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: a2a-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  POSTGRES_PASSWORD: $(echo -n "$POSTGRES_PASSWORD" | base64 -w 0)
  REDIS_PASSWORD: $(echo -n "$REDIS_PASSWORD" | base64 -w 0)
  A2A_SECRET_KEY: $(echo -n "$A2A_SECRET_KEY" | base64 -w 0)
  A2A_ADMIN_API_KEY: $(echo -n "$A2A_ADMIN_API_KEY" | base64 -w 0)
  A2A_SYSTEM_API_KEY: $(echo -n "$A2A_SYSTEM_API_KEY" | base64 -w 0)
EOF
    
    kubectl apply -f /tmp/a2a-secrets.yaml
    rm /tmp/a2a-secrets.yaml
    
    success "Production secrets generated and applied"
    
    # Save credentials securely
    cat > production-credentials.txt << EOF
# A2A Production Credentials - KEEP SECURE
# Generated: $(date)

POSTGRES_PASSWORD=$POSTGRES_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
A2A_SECRET_KEY=$A2A_SECRET_KEY
A2A_ADMIN_API_KEY=$A2A_ADMIN_API_KEY
A2A_SYSTEM_API_KEY=$A2A_SYSTEM_API_KEY

# API Endpoints
A2A_BASE_URL=https://api.timpayne.net/a2a
A2A_HEALTH_URL=https://api.timpayne.net/a2a/health
A2A_METRICS_URL=https://api.timpayne.net/a2a/metrics
EOF
    
    chmod 600 production-credentials.txt
    warning "Production credentials saved to production-credentials.txt - KEEP SECURE!"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log "Deploying A2A system to Kubernetes..."
    
    # Apply all manifests
    kubectl apply -f k8s/a2a-deployment.yaml
    
    success "Kubernetes manifests applied"
    
    # Wait for deployment to be ready
    log "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=$TIMEOUT deployment/$DEPLOYMENT_NAME -n $NAMESPACE
    
    success "Deployment is ready"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check pods
    log "Checking pod status..."
    kubectl get pods -n $NAMESPACE -l app=a2a-server
    
    # Check if all pods are running
    RUNNING_PODS=$(kubectl get pods -n $NAMESPACE -l app=a2a-server --field-selector=status.phase=Running --no-headers | wc -l)
    EXPECTED_PODS=3
    
    if [ "$RUNNING_PODS" -eq "$EXPECTED_PODS" ]; then
        success "$RUNNING_PODS/$EXPECTED_PODS pods are running"
    else
        error "Only $RUNNING_PODS/$EXPECTED_PODS pods are running"
    fi
    
    # Check services
    log "Checking service status..."
    kubectl get services -n $NAMESPACE
    
    # Check ingress
    log "Checking ingress status..."
    kubectl get ingress -n $NAMESPACE
    
    success "Deployment verification completed"
}

# Test endpoints
test_endpoints() {
    log "Testing production endpoints..."
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$SERVICE_IP" ]; then
        SERVICE_IP=$(kubectl get service $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
        warning "Using cluster IP: $SERVICE_IP"
    else
        success "Load balancer IP: $SERVICE_IP"
    fi
    
    # Test health endpoint
    log "Testing health endpoint..."
    if kubectl exec -n $NAMESPACE deployment/$DEPLOYMENT_NAME -- curl -f http://localhost:8080/health > /dev/null 2>&1; then
        success "Health endpoint is responding"
    else
        error "Health endpoint is not responding"
    fi
    
    # Test agent discovery
    log "Testing agent discovery endpoint..."
    if kubectl exec -n $NAMESPACE deployment/$DEPLOYMENT_NAME -- curl -f http://localhost:8080/.well-known/agent.json > /dev/null 2>&1; then
        success "Agent discovery endpoint is responding"
    else
        error "Agent discovery endpoint is not responding"
    fi
    
    success "Endpoint testing completed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up production monitoring..."
    
    # Create ServiceMonitor for Prometheus
    cat > /tmp/a2a-servicemonitor.yaml << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: a2a-metrics
  namespace: $NAMESPACE
  labels:
    app: a2a-server
spec:
  selector:
    matchLabels:
      app: a2a-server
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
EOF
    
    kubectl apply -f /tmp/a2a-servicemonitor.yaml 2>/dev/null || warning "ServiceMonitor not applied (Prometheus operator may not be installed)"
    rm /tmp/a2a-servicemonitor.yaml
    
    success "Monitoring setup completed"
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    cat > production-deployment-report.md << EOF
# A2A Production Deployment Report

**Deployment Date**: $(date)
**Namespace**: $NAMESPACE
**Cluster**: $(kubectl config current-context)

## Deployment Status

### Pods
\`\`\`
$(kubectl get pods -n $NAMESPACE -l app=a2a-server)
\`\`\`

### Services
\`\`\`
$(kubectl get services -n $NAMESPACE)
\`\`\`

### Ingress
\`\`\`
$(kubectl get ingress -n $NAMESPACE)
\`\`\`

### Resource Usage
\`\`\`
$(kubectl top pods -n $NAMESPACE 2>/dev/null || echo "Metrics server not available")
\`\`\`

## Endpoints
- Health Check: http://$SERVICE_IP:8080/health
- Agent Discovery: http://$SERVICE_IP:8080/.well-known/agent.json
- Metrics: http://$SERVICE_IP:9090/metrics

## Credentials
See production-credentials.txt (keep secure)

## Next Steps
1. Configure DNS for production domain
2. Set up SSL certificates
3. Configure monitoring dashboards
4. Run production load tests
EOF
    
    success "Deployment report generated: production-deployment-report.md"
}

# Main deployment function
main() {
    echo "🚀 A2A PRODUCTION DEPLOYMENT"
    echo "================================"
    
    check_prerequisites
    create_namespace
    generate_secrets
    deploy_to_kubernetes
    verify_deployment
    test_endpoints
    setup_monitoring
    generate_report
    
    echo ""
    echo "🎉 A2A PRODUCTION DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "================================"
    echo "✅ Namespace: $NAMESPACE"
    echo "✅ Pods: 3/3 running"
    echo "✅ Services: Active"
    echo "✅ Endpoints: Responding"
    echo "✅ Monitoring: Configured"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Review production-deployment-report.md"
    echo "2. Secure production-credentials.txt"
    echo "3. Configure DNS and SSL"
    echo "4. Run production validation tests"
    echo ""
    echo "🔗 Access URLs:"
    echo "   Health: http://$SERVICE_IP:8080/health"
    echo "   API: http://$SERVICE_IP:8080/"
    echo "   Metrics: http://$SERVICE_IP:9090/metrics"
}

# Run main function
main "$@"
