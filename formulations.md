# Mathematical Formulations in the HaleLab Federated SSL Project

## 1. Introduction

This document provides a detailed explanation of the core mathematical and algorithmic formulations used in this federated self-supervised learning project. It covers the loss functions for the self-supervised tasks and the federated aggregation mechanism.

## 2. Self-Supervised Learning (SSL) Tasks

Self-supervised learning is a key component of this project, enabling the model to learn meaningful feature representations from unlabeled data. We use two primary SSL tasks: Contrastive Learning and Rotation Prediction.

### 2.1. Contrastive Learning (SimCLR)

The goal of contrastive learning is to learn representations by pulling similar samples (positive pairs) closer together in the embedding space while pushing dissimilar samples (negative pairs) further apart. In our implementation, which is inspired by the SimCLR framework, a positive pair is created by generating two different augmented "views" of the same input image.

The loss function used for this task is the **Normalized Temperature-scaled Cross-Entropy (NT-Xent) Loss**. For a positive pair of augmented examples, $(i, j)$, the loss is defined as:

$$ 
l_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)} $$

where:
-   $z_i$ and $z_j$ are the feature representations of the two augmented views.
-   $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ is the cosine similarity between two feature vectors.
-   $\tau$ is a temperature parameter that scales the similarity scores.
-   $N$ is the number of samples in the batch. The summation in the denominator is over all $2N$ samples in the batch (including positive and negative pairs), excluding the sample $i$ itself.
-   $\mathbb{1}_{k \neq i}$ is an indicator function that is 1 if $k \neq i$ and 0 otherwise.

The total loss is computed over all positive pairs in the batch.

### 2.2. Rotation Prediction

The rotation prediction task is a simpler form of self-supervised learning where the model is trained to predict the rotation that has been applied to an input image. In our project, we apply one of four rotation angles to each image: 0°, 90°, 180°, or 270°.

This task is framed as a 4-class classification problem. The model must predict which of the four rotation angles was applied to the input image. The loss function used for this task is the standard **Cross-Entropy Loss**.

The Cross-Entropy Loss for a single sample is defined as:

$$ 
\text{CE Loss} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c}) 
$$

where:
-   $M$ is the number of classes (in this case, $M=4$).
-   $y_{o,c}$ is a binary indicator (1 if class `c` is the correct class for observation `o`, and 0 otherwise).
-   $p_{o,c}$ is the predicted probability that observation `o` belongs to class `c`.

## 3. Federated Learning Aggregation

Federated learning allows multiple clients to collaboratively train a model without sharing their raw data. The server coordinates the training process and aggregates the model updates from the clients.

### 3.1. Federated Averaging (FedAvg)

This project uses the **Federated Averaging (FedAvg)** algorithm for model aggregation. In this approach, the server sends the current global model to a set of clients. Each client then trains the model on its local data for a few epochs. The updated model parameters from each client are then sent back to the server.

The server aggregates the client models by computing a weighted average of their parameters. The weight for each client is proportional to the number of samples in its local dataset.

The formula for the FedAvg update is:

$$ 
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{t+1}^k 
$$

where:
-   $w_{t+1}$ are the parameters of the global model at round $t+1$.
-   $K$ is the number of clients participating in the round.
-   $n_k$ is the number of data samples on client $k$.
-   $n = \sum_{k=1}^{K} n_k$ is the total number of data samples across all participating clients.
-   $w_{t+1}^k$ are the model parameters received from client $k$ after its local training in round $t+1$.

This process is repeated for several rounds, leading to a global model that has learned from the data of all clients without any data leaving the client devices.
