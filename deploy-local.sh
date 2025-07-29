#!/bin/bash

# Local Docker Deployment for HaleLab Federated Learning
# This script deploys the federated learning system locally using Docker Compose

set -e

# ===========================================
# Configuration
# ===========================================
KAGGLE_USERNAME="shashwatchaturvedi35"
KAGGLE_KEY="e9f467dac1413fcbb0f1f6a954479b8c"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ===========================================
# Helper Functions
# ===========================================
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "\n${BLUE}$1${NC}"
    echo -e "${BLUE}$(echo $1 | sed 's/./=/g')${NC}"
}

# Setup environment
setup_environment() {
    print_header "Setting up Environment"
    
    # Create .env file for docker-compose
    cat > .env << EOF
KAGGLE_USERNAME=$KAGGLE_USERNAME
KAGGLE_KEY=$KAGGLE_KEY
EOF
    
    # Create logs directory
    mkdir -p logs
    
    print_status "Environment setup complete"
}

# Build Docker image
build_image() {
    print_header "Building Docker Image"
    
    print_info "Building halelab-federated-ssl image..."
    docker build -t halelab-federated-ssl .
    
    print_status "Docker image built successfully"
}

# Deploy using Docker Compose
deploy_local() {
    print_header "Deploying Federated Learning System"
    
    print_info "Starting services with Docker Compose..."
    docker-compose up -d
    
    print_status "Services started"
    
    # Wait a moment for services to initialize
    sleep 10
    
    print_info "Service status:"
    docker-compose ps
}

# Monitor deployment
monitor_deployment() {
    print_header "Monitoring Deployment"
    
    print_info "Container status:"
    docker-compose ps
    
    echo ""
    print_info "To view logs:"
    echo "All services: docker-compose logs -f"
    echo "Server only: docker-compose logs -f fl-server"
    echo "Client 1 only: docker-compose logs -f fl-client-1"
    echo "Client 2 only: docker-compose logs -f fl-client-2"
    
    echo ""
    print_info "Recent server logs:"
    docker-compose logs --tail=10 fl-server
}

# Show logs
show_logs() {
    print_header "📋 Service Logs"
    
    echo "=== Server Logs ==="
    docker-compose logs --tail=20 fl-server
    
    echo -e "\n=== Client 1 Logs ==="
    docker-compose logs --tail=20 fl-client-1
    
    echo -e "\n=== Client 2 Logs ==="
    docker-compose logs --tail=20 fl-client-2
}

# Cleanup function
cleanup() {
    print_header "Cleanup"
    
    print_info "Stopping and removing containers..."
    docker-compose down -v
    
    print_info "Removing Docker image..."
    docker rmi halelab-federated-ssl || true
    
    print_info "Cleaning up environment files..."
    rm -f .env
    
    print_status "Cleanup complete"
}

# Main execution
main() {
    print_header "🚀 Deploying HaleLab Federated SSL Learning Locally"
    
    setup_environment
    build_image
    deploy_local
    
    print_header "🎯 Local Deployment Complete"
    monitor_deployment
    
    echo ""
    print_header "📋 Next Steps"
    print_info "1. Monitor the federated learning progress: docker-compose logs -f"
    print_info "2. The experiment will run for 5 rounds with 2 clients"
    print_info "3. Client 1 performs Rotation SSL, Client 2 performs Contrastive SSL"
    print_info "4. Check server at: http://localhost:8080"
    print_info "5. Use './deploy-local.sh logs' to view recent logs"
    print_info "6. Use './deploy-local.sh cleanup' to stop and remove everything"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        cleanup
        ;;
    "logs")
        show_logs
        ;;
    "status")
        print_header "📊 Deployment Status"
        docker-compose ps
        ;;
    "follow")
        print_header "📋 Following All Logs"
        docker-compose logs -f
        ;;
    *)
        echo "Usage: $0 [deploy|cleanup|logs|status|follow]"
        echo "  deploy  - Deploy the federated learning system locally (default)"
        echo "  cleanup - Stop and remove all containers and images"
        echo "  logs    - Show recent logs from all services"
        echo "  status  - Show status of all containers"
        echo "  follow  - Follow logs from all services in real-time"
        exit 1
        ;;
esac
