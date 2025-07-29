"""haleLab: Federated SSL for Medical Imaging."""

# Make key components available at package level
from .task import (
    get_ssl_model, get_ssl_weights, set_ssl_weights,
    train_ssl, evaluate_ssl, load_ssl_data
)

from .ssl_models import (
    DownstreamClassificationModel, ContrastiveSSLModel, 
    RotationSSLModel, JigsawSSLModel
)

from .ssl_utils import (
    create_ssl_dataloader, create_downstream_dataloader, partition_data
)

__all__ = [
    'get_ssl_model', 'get_ssl_weights', 'set_ssl_weights',
    'train_ssl', 'evaluate_ssl', 'load_ssl_data',
    'DownstreamClassificationModel', 'ContrastiveSSLModel',
    'RotationSSLModel', 'JigsawSSLModel',
    'create_ssl_dataloader', 'create_downstream_dataloader', 'partition_data'
]
