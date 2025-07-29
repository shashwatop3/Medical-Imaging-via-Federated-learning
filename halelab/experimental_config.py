from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import torch


@dataclass
class DatasetConfig:
    name: str
    labeled_ratio: float
    unlabeled_ratio: float
    preprocessing: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        assert abs(self.labeled_ratio + self.unlabeled_ratio - 1.0) < 1e-6, \
            "Labeled and unlabeled ratios must sum to 1.0"


@dataclass
class ClientConfig:
    client_id: int
    ssl_task: str
    dataset_config: DatasetConfig
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    batch_size: int = 16
    
    def __str__(self):
        return (f"Client {self.client_id}: {self.ssl_task} task, "
                f"{self.dataset_config.name} dataset, "
                f"{self.dataset_config.labeled_ratio:.0%} labeled, "
                f"preprocessing: {self.dataset_config.preprocessing}")


@dataclass
class FederatedConfig:
    aggregation_strategy: str = "MultiTaskSSL"
    num_rounds: int = 3
    local_epochs: int = 10
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_available_clients: int = 2
    
    def __str__(self):
        return (f"{self.aggregation_strategy} strategy, "
                f"{self.num_rounds} rounds, "
                f"{self.local_epochs} local epochs")


class ExperimentalConfigurator:
    
    @staticmethod
    def create_heterogeneous_clients() -> List[ClientConfig]:
        
        client1_dataset = DatasetConfig(
            name="HAM10000",
            labeled_ratio=0.7,
            unlabeled_ratio=0.3,
            preprocessing=["resize", "artifact_removal"]
        )
        client1 = ClientConfig(
            client_id=1,
            ssl_task="rotation",
            dataset_config=client1_dataset
        )
        
        client2_dataset = DatasetConfig(
            name="HAM10000",
            labeled_ratio=0.5,
            unlabeled_ratio=0.5,
            preprocessing=["resize", "artifact_removal"]
        )
        client2 = ClientConfig(
            client_id=2,
            ssl_task="contrastive",
            dataset_config=client2_dataset
        )
        
        return [client1, client2]
    
    @staticmethod
    def create_federated_config() -> FederatedConfig:
        return FederatedConfig(
            aggregation_strategy="MultiTaskSSL",
            num_rounds=3,
            local_epochs=10,
            min_available_clients=2
        )
    
    @staticmethod
    def print_experimental_plan(client_configs: List[ClientConfig]):
        print("=" * 60)
        print("FEDERATED SSL LEARNING - 10 ROUNDS")
        print("=" * 60)
        
        print("\nCLIENT CONFIGURATIONS:")
        print("-" * 40)
        for config in client_configs:
            print(f"   {config}")
        
        fed_config = ExperimentalConfigurator.create_federated_config()
        print(f"\nFEDERATED SETUP:")
        print("-" * 40)
        print(f"   {fed_config}")
        
        print("=" * 60)


if __name__ == "__main__":
    client_configs = ExperimentalConfigurator.create_heterogeneous_clients()
    
    ExperimentalConfigurator.print_experimental_plan(client_configs)