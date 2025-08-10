# Architecture Diagrams

## 1. Model Architecture

This diagram illustrates the neural network architectures used in the project. It shows the shared ResNet50 backbone, the different self-supervised learning (SSL) heads that are trained on the clients, and the downstream classification model. The parts of the model that are aggregated on the server during federated learning are highlighted.

```mermaid
graph TD
    subgraph "Base Model"
        A[Input Image] --> B(ResNet50 Backbone);
        B --> B_out[2048 features];
    end

    subgraph "SSL Models (one of these is used on a client)"
        direction LR
        subgraph "Contrastive SSL"
            B_out --> C1(Shared Head) --> C2(Contrastive Head) --> C3(Normalized Projections);
        end
        subgraph "Rotation SSL"
            B_out --> D1(Shared Head) --> D2(Rotation Head) --> D3(Rotation Logits);
        end
    end

    subgraph "Downstream Model"
        F1(Classifier Head) --> F2(Class Logits);
    end

    C1 -- "if Contrastive" --> F1
    D1 -- "if Rotation" --> F1


    subgraph "Federated Parameters"
        style C1 fill:#f9f,stroke:#333,stroke-width:2px
        style D1 fill:#f9f,stroke:#333,stroke-width:2px
    end
```

## 2. Federated Learning Architecture

This diagram shows the federated learning setup. It depicts the central Flower server and multiple clients, each with its own SSL task. The server coordinates the training process, sending global model parameters to the clients and receiving updated parameters back for aggregation.

```mermaid
graph TD
    subgraph "Federated Learning System"
        Server[Flower Server]
        subgraph "Clients"
            Client1[Client 1: Rotation Task]
            Client2[Client 2: Contrastive Task]
            ClientN[Client N: ...]
        end
    end

    Server -->|Send Parameters| Client1
    Server -->|Send Parameters| Client2
    Server -->|Send Parameters| ClientN

    Client1 -->|Local Training| Client1
    Client2 -->|Local Training| Client2
    ClientN -->|Local Training| ClientN

    Client1 -->|Send Updates| Server
    Client2 -->|Send Updates| Server
    ClientN -->|Send Updates| Server

    Server -->|Aggregate| Server
```

## 3. Workflow Diagram

This diagram provides a high-level overview of the entire workflow, from data preparation and federated training to model evaluation and deployment.

```mermaid
graph TD
    A[Start] --> B{Data Preparation};
    B --> B1[Load HAM10000 Dataset];
    B1 --> B2[Create SSL DataLoaders for each client];
    B2 --> C{Federated SSL Training};
    C --> C1[Server initializes global model];
    C1 --> C2[Server selects clients for a round];
    C2 --> C3[Selected clients receive global model];
    C3 --> C4[Clients train model on their local SSL task];
    C4 --> C5[Clients send updated model back to server];
    C5 --> C6[Server aggregates client models];
    C6 --> C7{Repeat for N rounds};
    C7 -- Yes --> C2;
    C7 -- No --> D{Downstream Task Evaluation};
    D --> D1[Load trained SSL model backbone];
    D1 --> D2[Train a classifier on top of the frozen backbone];
    D2 --> D3[Evaluate classifier on test data];
    D3 --> E{Deployment};
    E --> E1[Deploy server and clients on Kubernetes];
    E1 --> F[End];
```

## 4. Kubernetes Architecture Diagram

This diagram shows how the different components of the system are deployed on a Google Kubernetes Engine (GKE) cluster. It illustrates the server and client deployments, the services for communication, and the use of secrets for configuration.

```mermaid
graph TD
    subgraph "Google Kubernetes Engine (GKE) Cluster"
        subgraph "Node 1"
            ServerDeployment[Server Deployment]
            ServerPod[Pod: Flower Server]
            ServerDeployment --> ServerPod
        end

        subgraph "Node 2"
            Client1Deployment[Client 1 Deployment]
            Client1Pod[Pod: FL Client 1]
            Client1Deployment --> Client1Pod
        end

        subgraph "Node 3"
            Client2Deployment[Client 2 Deployment]
            Client2Pod[Pod: FL Client 2]
            Client2Deployment --> Client2Pod
        end

        subgraph "Networking"
            ServerService["Server Service (LoadBalancer)"]
            ServerService --> ServerPod
        end

        subgraph "Configuration"
            KaggleSecret[Kaggle Secret]
            ConfigMap[ConfigMap]
        end
    end

    Client1Pod -- "Connects to" --> ServerService
    Client2Pod -- "Connects to" --> ServerService

    ServerPod -- "Uses" --> KaggleSecret
    Client1Pod -- "Uses" --> KaggleSecret
    Client2Pod -- "Uses" --> KaggleSecret

    ServerPod -- "Uses" --> ConfigMap
    Client1Pod -- "Uses" --> ConfigMap
    Client2Pod -- "Uses" --> ConfigMap
```
