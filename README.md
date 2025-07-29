# Federated Self-Supervised Learning for Medical Imaging

This project implements a federated learning system for training self-supervised learning (SSL) models on medical imaging data, specifically using the HAM10000 dataset. The system is designed to handle heterogeneous clients with different SSL tasks and data distributions, simulating a realistic federated learning scenario.

## Project Overview

The core idea is to leverage federated learning to train a powerful image representation model (a ResNet-50 backbone) on decentralized medical data without compromising data privacy. Clients train their models locally on their private data using different SSL tasks (e.g., Rotation Prediction, Contrastive Learning). Only the model updates (specifically, the parameters of a shared projection head) are sent to a central server for aggregation. This aggregated model can then be used for various downstream tasks, such as skin lesion classification.

## Key Features

*   **Federated Learning:** Utilizes the Flower framework for federated learning simulation and deployment.
*   **Self-Supervised Learning:** Implements multiple SSL tasks to learn representations from unlabeled medical images.
    *   **Rotation Prediction:** The model learns to predict the rotation applied to an image (0, 90, 180, or 270 degrees).
    *   **Contrastive Learning (SimCLR-style):** The model learns to pull augmented views of the same image closer together in the embedding space while pushing views from different images apart.
*   **Heterogeneous Clients:** The system is designed to support clients with different SSL tasks, data preprocessing pipelines, and data distributions.
*   **Kubernetes Deployment:** Includes Kubernetes manifests for deploying the federated learning server and clients in a containerized environment.
*   **Modular Architecture:** The code is structured into logical components for models, data utilities, federated strategies, and application logic.

## Architecture

The system follows a standard client-server architecture for federated learning:

1.  **Federated Server:** Responsible for orchestrating the training process. It sends the global model to the clients, waits for their updates, aggregates the updates to produce a new global model, and repeats the process for a specified number of rounds.
2.  **Federated Clients:** Each client represents a data silo (e.g., a hospital). It receives the global model from the server, trains it on its local data using its specific SSL task, and sends the updated model parameters back to the server.

The entire system is designed to be deployed on Kubernetes, ensuring scalability and manageability.

## Directory Structure

The project is organized as follows:

```
/
├── halelab/            # Main Python package for the application
│   ├── __init__.py
│   ├── client_app.py   # Flower client application logic
│   ├── server_app.py   # Flower server application logic
│   ├── task.py         # Defines model creation, training, and evaluation logic
│   ├── ssl_models.py   # PyTorch models for SSL tasks and downstream classification
│   ├── ssl_utils.py    # Data transformation and utility functions for SSL
│   ├── ham10000_utils.py # Data loading and management for the HAM10000 dataset
│   ├── experimental_config.py # Configuration for clients and federated setup
│   ├── federated_ssl_strategy.py # Custom Flower strategies for SSL aggregation
│   └── logging_config.py # Logging configuration
│
├── k8s/                # Kubernetes manifests for deployment
│   ├── namespace.yaml
│   ├── config.yaml
│   ├── kaggle-secret.yaml
│   ├── server-deployment.yaml
│   ├── client1-deployment.yaml
│   └── client2-deployment.yaml
│
├── deploy-local.sh     # Script to run the system locally
├── deploy-gke.sh       # Script to deploy the system on Google Kubernetes Engine
└── .gitignore          # Git ignore file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shashwatop3/Medical-Imaging-via-Federated-learning.git
    cd Medical-Imaging-via-Federated-learning
    ```

2.  **Create a Python virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file will need to be generated based on the project's imports).*

3.  **Kaggle API Credentials:**
    The data loader uses the Kaggle API to download the HAM10000 dataset. You need to have your Kaggle API key. The Kubernetes deployment uses a secret (`k8s/kaggle-secret.yaml`) to manage these credentials. For local execution, ensure your `kaggle.json` is in `~/.kaggle/`.

## How to Run

### Local Deployment

The `deploy-local.sh` script can be used to simulate the federated learning process on a single machine. It starts the server and two clients in the background.

```bash
bash deploy-local.sh
```

This will create log files (`server.log`, `client1.log`, `client2.log`) in the `logs/` directory.

### Kubernetes Deployment

The `deploy-gke.sh` script is provided to deploy the system to a Google Kubernetes Engine (GKE) cluster.

1.  **Prerequisites:**
    *   A GKE cluster.
    *   `gcloud` CLI configured to connect to your cluster.
    *   A Google Container Registry (GCR) repository to host the Docker image.

2.  **Build and Push the Docker Image:**
    You will need a `Dockerfile` to build the application image and push it to your GCR repository. The image path in the `.yaml` files (`gcr.io/moonlit-oven-464819-k8/halelab-federated-ssl`) will need to be updated to point to your repository.

3.  **Deploy to GKE:**
    ```bash
    bash deploy-gke.sh
    ```
    This script applies the Kubernetes manifests in the `k8s/` directory to deploy the server and clients.

## Self-Supervised Learning Implementation

The project implements a frozen ResNet-50 backbone for feature extraction. Only the SSL task-specific heads (and a shared projection head) are trained.

*   **Rotation Prediction:** A simple classification head is added to predict one of four rotation angles. The loss is calculated using Cross-Entropy Loss.
*   **Contrastive Learning:** A projection head is added to map features to an embedding space. The NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss is used to maximize agreement between different augmentations of the same image.

The `MultiTaskSSLStrategy` in `federated_ssl_strategy.py` is a custom Flower strategy designed to handle the aggregation of models trained on different SSL tasks, although the current implementation uses a simplified aggregation approach.
