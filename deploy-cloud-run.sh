#!/bin/bash

# Google Cloud Run Deployment for HaleLab Federated Learning
# This script deploys the federated learning system using Docker containers on Google Cloud Run

set -e

# ===========================================
# Configuration
# ===========================================
PROJECT_ID="moonlit-oven-464819-k8"
REGION="us-central1"
SERVICE_NAME="halelab-fl"
IMAGE_NAME="gcr.io/$PROJECT_ID/halelab-federated-ssl"
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

# Setup Google Cloud project
setup_gcp_project() {
    print_header "Setting up Google Cloud Project"
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_warning "No active gcloud authentication found"
        print_info "Please authenticate with: gcloud auth login"
        print_info "Then run this script again"
        exit 1
    fi
    
    print_info "Authenticated as: $(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1)"
    
    gcloud config set project $PROJECT_ID
    gcloud config set run/region $REGION
    
    # Configure Docker for GCR
    print_info "Configuring Docker for Google Container Registry..."
    if ! echo 'y' | gcloud auth configure-docker --quiet; then
        print_error "Failed to configure Docker authentication"
        exit 1
    fi
    
    # Enable required APIs
    print_info "Enabling required APIs..."
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    gcloud services enable logging.googleapis.com
    gcloud services enable monitoring.googleapis.com
    
    print_status "Google Cloud project configured"
}

# Build and push Docker image
build_and_push_image() {
    print_header "Building and Pushing Docker Image"
    
    print_info "Building Docker image for linux/amd64..."
    docker buildx build --platform linux/amd64 -t $IMAGE_NAME --push .
    
    print_status "Docker image built and pushed"
}

# Deploy server
deploy_server() {
    print_header "Deploying FL Server to Cloud Run"
    
    print_info "Deploying server service..."
    gcloud run deploy ${SERVICE_NAME}-server \
        --image $IMAGE_NAME \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 2 \
        --min-instances 1 \
        --max-instances 1 \
        --timeout 900 \
        --set-env-vars "ROLE=server,HOST=0.0.0.0,NUM_ROUNDS=10,MIN_CLIENTS=2,LOG_LEVEL=INFO,STRATEGY_TYPE=MultiTaskSSL" \
        --command "python3" \
        --args "distributed_server.py,--port,8080,--host,0.0.0.0,--num-rounds,10,--min-clients,2"
    
    # Get server URL
    SERVER_URL=$(gcloud run services describe ${SERVICE_NAME}-server --region=$REGION --format="value(status.url)")
    print_status "Server deployed at: $SERVER_URL"
    
    # Extract hostname for clients
    SERVER_HOST=$(echo $SERVER_URL | sed 's|https://||' | sed 's|http://||')
    echo "SERVER_HOST=$SERVER_HOST" > .env
}

# Deploy clients
deploy_clients() {
    print_header "Deploying FL Clients to Cloud Run"
    
    # Source the server host
    source .env
    
    # Deploy Client 1 (Rotation SSL)
    print_info "Deploying client 1 (Rotation SSL)..."
    gcloud run deploy ${SERVICE_NAME}-client-1 \
        --image $IMAGE_NAME \
        --platform managed \
        --region $REGION \
        --no-allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --min-instances 1 \
        --max-instances 1 \
        --timeout 900 \
        --set-env-vars "ROLE=client,CLIENT_ID=1,SSL_TASK=rotation,SERVER_ADDRESS=$SERVER_HOST,LOG_LEVEL=INFO,KAGGLE_USERNAME=$KAGGLE_USERNAME,KAGGLE_KEY=$KAGGLE_KEY" \
        --command "python3" \
        --args "cloud_run_client.py"
    
    # Deploy Client 2 (Contrastive SSL)
    print_info "Deploying client 2 (Contrastive SSL)..."
    gcloud run deploy ${SERVICE_NAME}-client-2 \
        --image $IMAGE_NAME \
        --platform managed \
        --region $REGION \
        --no-allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --min-instances 1 \
        --max-instances 1 \
        --timeout 900 \
        --set-env-vars "ROLE=client,CLIENT_ID=2,SSL_TASK=contrastive,SERVER_ADDRESS=$SERVER_HOST,LOG_LEVEL=INFO,KAGGLE_USERNAME=$KAGGLE_USERNAME,KAGGLE_KEY=$KAGGLE_KEY" \
        --command "python3" \
        --args "cloud_run_client.py"
    
    print_status "Clients deployed"
}

# Monitor deployment
monitor_deployment() {
    print_header "Monitoring Deployment"
    
    print_info "Service URLs:"
    gcloud run services list --region=$REGION --filter="metadata.name:${SERVICE_NAME}" --format="table(metadata.name,status.url,status.conditions[0].type)"
    
    echo ""
    print_info "To view logs:"
    echo "Server: gcloud run services logs read ${SERVICE_NAME}-server --region=$REGION"
    echo "Client 1: gcloud run services logs read ${SERVICE_NAME}-client-1 --region=$REGION"
    echo "Client 2: gcloud run services logs read ${SERVICE_NAME}-client-2 --region=$REGION"
}

# Cleanup function
cleanup() {
    print_header "Cleanup (Optional)"
    print_info "To delete services, run:"
    echo "gcloud run services delete ${SERVICE_NAME}-server --region=$REGION"
    echo "gcloud run services delete ${SERVICE_NAME}-client-1 --region=$REGION"
    echo "gcloud run services delete ${SERVICE_NAME}-client-2 --region=$REGION"
}

# Main execution
main() {
    print_header "🚀 Deploying HaleLab Federated SSL Learning to Google Cloud Run"
    
    setup_gcp_project
    build_and_push_image
    deploy_server
    sleep 30  # Wait for server to be ready
    deploy_clients
    
    print_header "🎯 Deployment Complete"
    monitor_deployment
    
    echo ""
    print_header "📋 Next Steps"
    print_info "1. Monitor the federated learning progress using the log commands above"
    print_info "2. The experiment will run for 10 rounds with 2 clients"
    print_info "3. Client 1 performs Rotation SSL, Client 2 performs Contrastive SSL"
    
    echo ""
    cleanup
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "auth")
        print_header "🔐 Setting up Google Cloud Authentication"
        print_info "Opening web browser for authentication..."
        gcloud auth login
        echo 'y' | gcloud auth configure-docker
        print_status "Authentication complete. You can now run: $0 deploy"
        ;;
    "cleanup")
        print_header "🧹 Cleaning up Cloud Run services"
        gcloud run services delete ${SERVICE_NAME}-server --region=$REGION --quiet || true
        gcloud run services delete ${SERVICE_NAME}-client-1 --region=$REGION --quiet || true
        gcloud run services delete ${SERVICE_NAME}-client-2 --region=$REGION --quiet || true
        print_status "Cleanup complete"
        ;;
    "logs")
        print_header "📋 Fetching logs"
        echo "=== Server Logs ==="
        gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}-server" --limit=50 --format="table(timestamp,textPayload)"
        echo -e "\n=== Client 1 Logs ==="
        gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}-client-1" --limit=50 --format="table(timestamp,textPayload)"
        echo -e "\n=== Client 2 Logs ==="
        gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}-client-2" --limit=50 --format="table(timestamp,textPayload)"
        ;;
    "status")
        print_header "📊 Deployment Status"
        gcloud run services list --region=$REGION --filter="metadata.name:${SERVICE_NAME}" --format="table(metadata.name,status.url,status.conditions[0].type,spec.template.spec.containers[0].resources.limits)"
        ;;
    *)
        echo "Usage: $0 [deploy|auth|cleanup|logs|status]"
        echo "  deploy  - Deploy the federated learning system (default)"
        echo "  auth    - Set up Google Cloud authentication"
        echo "  cleanup - Remove all deployed services"
        echo "  logs    - Show recent logs from all services"
        echo "  status  - Show status of deployed services"
        exit 1
        ;;
esac
