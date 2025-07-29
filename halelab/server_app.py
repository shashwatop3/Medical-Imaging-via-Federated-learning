import flwr as fl
import numpy as np
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from .task import get_model, get_weights, get_ssl_model, get_ssl_weights
from .federated_ssl_strategy import create_ssl_strategy


def multi_task_ssl_server_fn():
    print("Starting Multi-Task SSL Federated Server")
    print("=" * 60)
    print("SSL Tasks:")
    print("   Client 1: Contrastive Learning (SimCLR)")
    print("   Client 2: Rotation Prediction")
    print("   Client 3: Jigsaw Puzzle Solving")
    print("=" * 60)
    
    strategy = create_ssl_strategy(
        evaluate_metrics_aggregation_fn=lambda metrics: {"accuracy": np.mean([m["accuracy"] for _, m in metrics])}
    )
    return strategy


def ssl_server_fn():
    ssl_task_type = "rotation"
    
    ssl_model = get_ssl_model(ssl_task_type, pretrained=True)
    ndarrays = get_ssl_weights(ssl_model)
    parameters = ndarrays_to_parameters(ndarrays)
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    return strategy


def server_fn():
    model = get_model()
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    return strategy