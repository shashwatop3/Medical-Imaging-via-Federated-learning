#!/bin/bash

# Google Kubernetes Engine Deployment for HaleLab Federated Learning
# This script deploys the federated learning system using Kubernetes on GKE

set -e

# ===========================================
# Configuration
# ===========================================
PROJECT_ID="moonlit-oven-464819-k8"
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="halelab-fl-cluster"
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
    
    gcloud config set project $PROJECT_ID
    gcloud config set compute/zone $ZONE
    gcloud config set compute/region $REGION
    
    # Enable required APIs
    print_info "Enabling required APIs..."
    gcloud services enable container.googleapis.com
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    gcloud services enable logging.googleapis.com
    gcloud services enable monitoring.googleapis.com
    
    print_status "Google Cloud project configured"
}

# Create GKE cluster
create_cluster() {
    print_header "Creating GKE Cluster"
    
    if gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE &> /dev/null; then
        print_warning "Cluster $CLUSTER_NAME already exists"
    else
        print_info "Creating GKE cluster..."
        gcloud container clusters create $CLUSTER_NAME \
            --zone=$ZONE \
            --num-nodes=3 \
            --machine-type=n1-standard-4 \
            --disk-size=100GB \
            --enable-autorepair \
            --enable-autoupgrade \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=5 \
            --logging=SYSTEM,WORKLOAD \
            --monitoring=SYSTEM
        
        print_status "GKE cluster created"
    fi
    
    # Get cluster credentials
    gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE
}

# Build and push Docker image
build_and_push_image() {
    print_header "Building and Pushing Docker Image"
    
    print_info "Building Docker image..."
    docker build --platform linux/amd64 -t $IMAGE_NAME .
    
    print_info "Pushing image to Google Container Registry..."
    docker push $IMAGE_NAME
    
    print_status "Docker image built and pushed"
}

# Create Kubernetes manifests
create_k8s_manifests() {
    print_header "Creating Kubernetes Manifests"
    
    mkdir -p k8s
    
    # Create namespace
    cat > k8s/namespace.yaml << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: halelab-fl
  labels:
    name: halelab-fl
EOF

    # Create secret for Kaggle credentials
    cat > k8s/kaggle-secret.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: kaggle-credentials
  namespace: halelab-fl
type: Opaque
stringData:
  username: "$KAGGLE_USERNAME"
  key: "$KAGGLE_KEY"
EOF

    # Create ConfigMap for application configuration
    cat > k8s/config.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: fl-config
  namespace: halelab-fl
data:
  NUM_ROUNDS: "10"
  MIN_CLIENTS: "2"
  LOG_LEVEL: "INFO"
  STRATEGY_TYPE: "MultiTaskSSL"
EOF

    # Create Server Deployment
    cat > k8s/server-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fl-server
  namespace: halelab-fl
  labels:
    app: fl-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fl-server
  template:
    metadata:
      labels:
        app: fl-server
    spec:
      containers:
      - name: fl-server
        image: $IMAGE_NAME
        ports:
        - containerPort: 8080
          name: flower-port
        env:
        - name: ROLE
          value: "server"
        - name: PORT
          value: "8080"
        - name: HOST
          value: "0.0.0.0"
        envFrom:
        - configMapRef:
            name: fl-config
        command: ["python3"]
        args: ["distributed_server.py", "--port", "8080", "--host", "0.0.0.0", "--num-rounds", "10", "--min-clients", "2"]
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
            ephemeral-storage: "20Gi"
          limits:
            memory: "2Gi"
            cpu: "2000m"
            ephemeral-storage: "50Gi"
        livenessProbe:
          tcpSocket:
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          tcpSocket:
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: fl-server-service
  namespace: halelab-fl
spec:
  selector:
    app: fl-server
  ports:
  - port: 8080
    targetPort: 8080
    name: flower-port
  type: ClusterIP
EOF

    # Create Client 1 Deployment (Rotation SSL)
    cat > k8s/client1-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fl-client-1
  namespace: halelab-fl
  labels:
    app: fl-client-1
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fl-client-1
  template:
    metadata:
      labels:
        app: fl-client-1
    spec:
      containers:
      - name: fl-client-1
        image: $IMAGE_NAME
        env:
        - name: ROLE
          value: "client"
        - name: CLIENT_ID
          value: "1"
        - name: SSL_TASK
          value: "rotation"
        - name: SERVER_ADDRESS
          value: "fl-server-service.halelab-fl.svc.cluster.local:8080"
        - name: KAGGLE_USERNAME
          valueFrom:
            secretKeyRef:
              name: kaggle-credentials
              key: username
        - name: KAGGLE_KEY
          valueFrom:
            secretKeyRef:
              name: kaggle-credentials
              key: key
        envFrom:
        - configMapRef:
            name: fl-config
        command: ["python3"]
        args: ["distributed_client.py", "--server-address", "fl-server-service.halelab-fl.svc.cluster.local:8080", "--client-id", "1", "--ssl-task", "rotation"]
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
            ephemeral-storage: "20Gi"
          limits:
            memory: "2Gi"
            cpu: "2000m"
            ephemeral-storage: "50Gi"
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: kaggle-config
          mountPath: /root/.kaggle
      volumes:
      - name: logs
        emptyDir: {}
      - name: kaggle-config
        emptyDir: {}
EOF

    # Create Client 2 Deployment (Contrastive SSL)
    cat > k8s/client2-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fl-client-2
  namespace: halelab-fl
  labels:
    app: fl-client-2
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fl-client-2
  template:
    metadata:
      labels:
        app: fl-client-2
    spec:
      containers:
      - name: fl-client-2
        image: $IMAGE_NAME
        env:
        - name: ROLE
          value: "client"
        - name: CLIENT_ID
          value: "2"
        - name: SSL_TASK
          value: "contrastive"
        - name: SERVER_ADDRESS
          value: "fl-server-service.halelab-fl.svc.cluster.local:8080"
        - name: KAGGLE_USERNAME
          valueFrom:
            secretKeyRef:
              name: kaggle-credentials
              key: username
        - name: KAGGLE_KEY
          valueFrom:
            secretKeyRef:
              name: kaggle-credentials
              key: key
        envFrom:
        - configMapRef:
            name: fl-config
        command: ["python3"]
        args: ["distributed_client.py", "--server-address", "fl-server-service.halelab-fl.svc.cluster.local:8080", "--client-id", "2", "--ssl-task", "contrastive"]
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
            ephemeral-storage: "20Gi"
          limits:
            memory: "2Gi"
            cpu: "2000m"
            ephemeral-storage: "50Gi"
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: kaggle-config
          mountPath: /root/.kaggle
      volumes:
      - name: logs
        emptyDir: {}
      - name: kaggle-config
        emptyDir: {}
EOF

    print_status "Kubernetes manifests created"
}

# Deploy to Kubernetes
deploy_to_k8s() {
    print_header "Deploying to Kubernetes"
    
    print_info "Applying Kubernetes manifests..."
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/kaggle-secret.yaml
    kubectl apply -f k8s/config.yaml
    kubectl apply -f k8s/server-deployment.yaml
    
    # Wait for server to be ready
    print_info "Waiting for server to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/fl-server -n halelab-fl
    
    # Deploy clients
    kubectl apply -f k8s/client1-deployment.yaml
    kubectl apply -f k8s/client2-deployment.yaml
    
    print_status "Deployed to Kubernetes"
}

# Monitor deployment
monitor_deployment() {
    print_header "Monitoring Deployment"
    
    print_info "Pod status:"
    kubectl get pods -n halelab-fl -o wide
    
    echo ""
    print_info "Service status:"
    kubectl get services -n halelab-fl
    
    echo ""
    print_info "Deployment status:"
    kubectl get deployments -n halelab-fl
    
    echo ""
    print_info "To view logs:"
    echo "Server: kubectl logs -f deployment/fl-server -n halelab-fl"
    echo "Client 1: kubectl logs -f deployment/fl-client-1 -n halelab-fl"
    echo "Client 2: kubectl logs -f deployment/fl-client-2 -n halelab-fl"
    
    echo ""
    print_info "To monitor all logs:"
    echo "kubectl logs -f -l app=fl-server -n halelab-fl"
}

# Cleanup function
cleanup() {
    print_header "Cleanup"
    print_info "Deleting Kubernetes resources..."
    kubectl delete namespace halelab-fl --ignore-not-found=true
    
    print_info "Deleting GKE cluster..."
    gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE --quiet
    
    print_status "Cleanup complete"
}

# Main execution
main() {
    print_header "🚀 Deploying HaleLab Federated SSL Learning to GKE"
    
    setup_gcp_project
    create_cluster
    build_and_push_image
    create_k8s_manifests
    deploy_to_k8s
    
    print_header "🎯 Deployment Complete"
    monitor_deployment
    
    echo ""
    print_header "📋 Next Steps"
    print_info "1. Monitor the federated learning progress using the log commands above"
    print_info "2. The experiment will run for 5 rounds with 2 clients"
    print_info "3. Client 1 performs Rotation SSL, Client 2 performs Contrastive SSL"
    print_info "4. Use 'kubectl get pods -n halelab-fl -w' to watch pod status"
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
        print_header "📋 Fetching logs"
        echo "=== Server Logs ==="
        kubectl logs deployment/fl-server -n halelab-fl --tail=50
        echo -e "\n=== Client 1 Logs ==="
        kubectl logs deployment/fl-client-1 -n halelab-fl --tail=50
        echo -e "\n=== Client 2 Logs ==="
        kubectl logs deployment/fl-client-2 -n halelab-fl --tail=50
        ;;
    "status")
        print_header "📊 Deployment Status"
        kubectl get all -n halelab-fl
        ;;
    *)
        echo "Usage: $0 [deploy|cleanup|logs|status]"
        echo "  deploy  - Deploy the federated learning system (default)"
        echo "  cleanup - Remove cluster and all resources"
        echo "  logs    - Show recent logs from all services"
        echo "  status  - Show status of all Kubernetes resources"
        exit 1
        ;;
esac
