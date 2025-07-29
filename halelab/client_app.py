import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import itertools
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import numpy as np
import flwr as fl

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import flwr as fl

from .experimental_config import ClientConfig, DatasetConfig
from .task import get_ssl_model, get_ssl_weights, set_ssl_weights, train_ssl, evaluate_ssl, get_weights, set_weights
from .ham10000_utils import HAM10000DataManager

logger = logging.getLogger(__name__)

print('[DEBUG] client_app.py module loaded', flush=True)

class SSLClient(fl.client.NumPyClient):
    
    def __init__(self, client_config: ClientConfig, optimizer_type: str = "Adam", 
                 ham10000_manager: Optional[HAM10000DataManager] = None):
        self.client_config = client_config
        self.optimizer_type = optimizer_type
        self.ham10000_manager = ham10000_manager
        self.metrics_log = []
        
        self._initialize_model_and_data()
        
        logger.info(f"SSL Client initialized for client {client_config.client_id}")
        logger.info(f"SSL Task: {client_config.ssl_task}")
        logger.info(f"Dataset: {client_config.dataset_config.name}")
    
    def _initialize_model_and_data(self):
        try:
            self.model = get_ssl_model(self.client_config.ssl_task)
            
            if self.ham10000_manager:
                self.train_loader, self.val_loader = self._create_data_loaders()
                self.labeled_loader = self._create_labeled_dataloader()
            else:
                logger.warning("No HAM10000 manager provided, using dummy data")
                self.train_loader, self.val_loader = self._create_dummy_loaders()
                self.labeled_loader = None
        
            self.optimizer = self._create_optimizer()
            
            logger.info("Model and data initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize model and data: {e}")
            raise
    
    def _create_data_loaders(self):
        from .ham10000_utils import create_ham10000_ssl_dataloader
        
        return create_ham10000_ssl_dataloader(
            data_manager=self.ham10000_manager,
            partition_id=self.client_config.client_id - 1,
            num_partitions=2,
            ssl_task=self.client_config.ssl_task,
            batch_size=self.client_config.batch_size or 8
        )
    
    def _create_labeled_dataloader(self):
        from .ham10000_utils import create_ham10000_downstream_dataloader
        from torch.utils.data import DataLoader
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        try:
            client_partitions = self.ham10000_manager.create_heterogeneous_client_partitions(2)
            partition_id = self.client_config.client_id - 1
            
            if partition_id not in client_partitions:
                return None
                
            image_paths, labels = client_partitions[partition_id]
            
            if labels is None:
                return None
            
            unique_classes = np.unique(labels)
            print(f"Available classes in client data: {unique_classes}")
            
            subset_size = min(500, len(image_paths))
            
            if len(unique_classes) > 1:
                image_paths_subset, _, labels_subset, _ = train_test_split(
                    image_paths, labels, 
                    train_size=subset_size, 
                    random_state=42, 
                    stratify=labels
                )
            else:
                image_paths_subset = image_paths[:subset_size]
                labels_subset = labels[:subset_size]
            
            print(f"Downstream training subset: {len(image_paths_subset)} samples")
            print(f"Class distribution: {np.bincount(labels_subset)}")
            
            labeled_dataset = self.ham10000_manager.create_downstream_dataset(image_paths_subset, labels_subset)
            
            labeled_loader = DataLoader(
                labeled_dataset,
                batch_size=16,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            
            print(f"Created labeled dataloader with {len(labeled_dataset)} samples for downstream training")
            return labeled_loader
            
        except Exception as e:
            logger.warning(f"Failed to create labeled dataloader: {e}")
            return None
    
    def _create_dummy_loaders(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        dummy_data = torch.randn(100, 3, 224, 224)
        dummy_labels = torch.randint(0, 4, (100,))
        
        if self.client_config.ssl_task == "contrastive":
            dataset = TensorDataset(dummy_data, dummy_data, dummy_labels)
        else:
            dataset = TensorDataset(dummy_data, dummy_labels)
        
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def _create_optimizer(self):
        if self.optimizer_type.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=self.client_config.learning_rate or 0.001)
        elif self.optimizer_type.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.client_config.learning_rate or 0.01)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
    
    def get_parameters(self, config):
        print('[DEBUG] get_parameters called', flush=True)
        print(f'[DEBUG] get_parameters - config: {config}', flush=True)
        print(f'[DEBUG] get_parameters - self.model type: {type(self.model)}', flush=True)
        logger.debug("get_parameters called")
        
        actual_params = get_weights(self.model)
        print(f'[DEBUG] get_parameters - returning actual parameters: {len(actual_params)}', flush=True)
        return actual_params
    
    def fit(self, parameters, config):
        print('[DEBUG] fit (module-level print) called', flush=True)
        print('[DEBUG] fit (very first line) called', flush=True)
        print(f'[DEBUG] fit() - parameters type: {type(parameters)}, length: {len(parameters) if hasattr(parameters, "__len__") else "N/A"}', flush=True)
        print(f'[DEBUG] fit() - config type: {type(config)}, content: {config}', flush=True)
        print(f'[DEBUG] fit() - self type: {type(self)}', flush=True)
        print(f'[DEBUG] fit() - self.client_config: {self.client_config}', flush=True)
        logger.info(f"[FIT] Entered fit() for client {self.client_config.client_id} | SSL task: {self.client_config.ssl_task}")
        try:
            print('[DEBUG] fit() - about to set model parameters', flush=True)
            set_weights(self.model, parameters)
            print('[DEBUG] fit() - model parameters set successfully', flush=True)
            
            epochs = config.get("epochs", 10) if config else 10
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f'[DEBUG] fit() - about to train for {epochs} epochs on {device}', flush=True)
            
            train_loss = train_ssl(
                model=self.model,
                dataloader=self.train_loader,
                epochs=epochs,
                device=device,
                task_type=self.client_config.ssl_task
            )
            print(f'[DEBUG] fit() - training completed with loss: {train_loss}', flush=True)
            
            updated_parameters = get_weights(self.model)
            print(f'[DEBUG] fit() - got updated parameters, length: {len(updated_parameters)}', flush=True)
            
            try:
                if hasattr(self.train_loader.dataset, '__len__'):
                    dataset_size = len(self.train_loader.dataset)
                else:
                    dataset_size = 100
            except (TypeError, AttributeError):
                dataset_size = 100
            print(f'[DEBUG] fit() - dataset size: {dataset_size}', flush=True)
            
            if train_loss is not None:
                logger.info(f"Training completed. Loss: {train_loss:.4f}")
            else:
                logger.info("Training completed. Loss: None")
            print(f'[DEBUG] fit() - returning results', flush=True)
            
            metrics = {
                "ssl_train_loss": train_loss,
                "ssl_task": self.client_config.ssl_task,
                "client_id": self.client_config.client_id,
                "preprocessing": ", ".join(self.client_config.dataset_config.preprocessing),
                "round": config.get("server_round", -1)
            }
            
            return updated_parameters, dataset_size, metrics
            
        except Exception as e:
            print(f'[DEBUG] fit() - EXCEPTION: {e}', flush=True)
            logger.error(f"Error during training: {e}")
            raise

    def _get_predictions(self) -> Tuple[List[int], List[int]]:
        from halelab.ssl_models import DownstreamClassificationModel
        
        print(f"🔍 Creating downstream classifier for Client {self.client_config.client_id} with SSL task: {self.client_config.ssl_task}")
        
        downstream_model = DownstreamClassificationModel(
            ssl_model=self.model, 
            num_classes=7,
            freeze_backbone=True
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        downstream_model.to(device)
        
        print(f"✅ Downstream model created successfully for {self.client_config.ssl_task} SSL task")
        
        if hasattr(self, 'labeled_loader') and self.labeled_loader:
            print(f"📚 Training downstream classifier for Client {self.client_config.client_id} ({self.client_config.ssl_task})...")
            downstream_model.train()
            
            DOWNSTREAM_LR = 0.001
            DOWNSTREAM_EPOCHS = 3
            DOWNSTREAM_MAX_BATCHES = 25
            
            optimizer = torch.optim.Adam(downstream_model.classifier.parameters(), lr=DOWNSTREAM_LR)
            
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            total_batches = 0
            
            print(f"   Using consistent parameters: LR={DOWNSTREAM_LR}, Epochs={DOWNSTREAM_EPOCHS}, MaxBatches={DOWNSTREAM_MAX_BATCHES}")
            
            for epoch in range(DOWNSTREAM_EPOCHS):
                epoch_loss = 0.0
                epoch_batches = 0
                for batch_idx, (data, labels) in enumerate(self.labeled_loader):
                    if batch_idx >= DOWNSTREAM_MAX_BATCHES:
                        break
                    data, labels = data.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = downstream_model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_batches += 1
                    total_loss += loss.item()
                    total_batches += 1
                
                if epoch_batches > 0:
                    avg_epoch_loss = epoch_loss / epoch_batches
                    print(f"   Downstream epoch {epoch+1}/{DOWNSTREAM_EPOCHS}: loss = {avg_epoch_loss:.4f}")
            
            if total_batches > 0:
                avg_total_loss = total_loss / total_batches
                print(f"✅ Downstream training completed. Average loss: {avg_total_loss:.4f}")
                print(f"   Total batches processed: {total_batches}")
            else:
                print("⚠️  No training data available for downstream classifier")
        
        print(f"🔍 Getting predictions for Client {self.client_config.client_id} ({self.client_config.ssl_task})...")
        downstream_model.eval()
        y_true = []
        y_pred = []
        total_samples = 0

        if hasattr(self, 'labeled_loader') and self.labeled_loader:
            print("   Using labeled dataloader for downstream evaluation")
            eval_loader = self.labeled_loader
        else:
            print("   No labeled dataloader available, using validation loader")
            eval_loader = self.val_loader
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                try:
                    if len(batch) == 2:
                        data, labels = batch
                    else:
                        raise ValueError(f"Unexpected batch format: {type(batch)}")
                    
                    if isinstance(data, (tuple, list)) and len(data) == 2:
                        data = data[0]
                        if batch_idx == 0:
                            print(f"   Detected multi-view data format (SSL task: {self.client_config.ssl_task})")
                            print(f"   Using first view - Data type: {type(data)}, Data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")
                    elif batch_idx == 0:
                        print(f"   Detected single-view data format (SSL task: {self.client_config.ssl_task})")
                        print(f"   Data type: {type(data)}, Data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")
                    
                    if not hasattr(data, 'to'):
                        raise ValueError(f"Data is not a tensor. Type: {type(data)}, Content preview: {str(data)[:100]}...")
                    
                    data, labels = data.to(device), labels.to(device)
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    print(f"Batch type: {type(batch)}, Batch length: {len(batch) if hasattr(batch, '__len__') else 'No length'}")
                    if len(batch) == 2:
                        data, labels = batch
                        print(f"Data type: {type(data)}, Labels type: {type(labels)}")
                        if isinstance(data, (tuple, list)):
                            print(f"Data is tuple/list with length: {len(data)}")
                            for i, item in enumerate(data):
                                print(f"  Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'No shape'}")
                    raise
                
                outputs = downstream_model(data)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                total_samples += labels.size(0)
                
                if batch_idx >= 20:
                    break
        
        print(f"✅ Predictions completed for {total_samples} samples")
        print(f"   Unique predictions: {set(y_pred)}")
        print(f"   Unique true labels: {set(y_true)}")
        return y_true, y_pred

    def _log_client_metrics(self, server_round: int, ssl_loss: float, ssl_accuracy: float, 
                            downstream_accuracy: float, precision: float, recall: float, f1_score: float,
                            notes: str = ""):
        metrics_data = {
            "Round": server_round,
            "ClientID": self.client_config.client_id,
            "Preprocessing": ", ".join(self.client_config.dataset_config.preprocessing),
            "SSL_Task": self.client_config.ssl_task,
            "SSL_Accuracy": ssl_accuracy,
            "SSL_Loss": ssl_loss,
            "Accuracy": downstream_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score,
            "Notes": notes
        }
        self.metrics_log.append(metrics_data)

        print("=" * 80)
        print(f"📊 CLIENT {self.client_config.client_id} METRICS - ROUND {server_round}")
        print("=" * 80)
        print(f"🔧 SSL Task: {self.client_config.ssl_task}")
        print(f"📈 SSL Accuracy: {ssl_accuracy:.4f}")
        print(f"📉 SSL Loss: {ssl_loss:.4f}")
        print("-" * 80)
        print("🎯 DOWNSTREAM CLASSIFICATION METRICS:")
        print(f"   • Accuracy:  {downstream_accuracy:.4f} ({downstream_accuracy*100:.2f}%)")
        print(f"   • Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   • Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   • F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
        print("=" * 80)
        print()

        log_df = pd.DataFrame(self.metrics_log)
        log_df.to_csv(f"client_{self.client_config.client_id}_metrics.csv", index=False)
    
    def evaluate(self, parameters, config):
        print('[DEBUG] evaluate called', flush=True)
        print(f'[DEBUG] evaluate - config: {config}', flush=True)
        logger.debug("evaluate called")
        try:
            set_weights(self.model, parameters)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            val_loss, val_metric = evaluate_ssl(
                model=self.model,
                dataloader=self.val_loader,
                device=device,
                task_type=self.client_config.ssl_task
            )

            y_true, y_pred = self._get_predictions()
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

            self._log_client_metrics(
                server_round=config.get("server_round", -1),
                ssl_loss=val_loss,
                ssl_accuracy=val_metric,
                downstream_accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1
            )
            
            try:
                if hasattr(self.val_loader.dataset, '__len__'):
                    dataset_size = len(self.val_loader.dataset)
                else:
                    dataset_size = 100
            except (TypeError, AttributeError):
                dataset_size = 100
            
            logger.info(f"Evaluation completed. Loss: {val_loss:.4f}, Metric: {val_metric:.4f}")
            
            return val_loss, dataset_size, {
                "ssl_accuracy": val_metric,
                "ssl_loss": val_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "ssl_task": self.client_config.ssl_task,
                "client_id": self.client_config.client_id,
                "preprocessing": ", ".join(self.client_config.dataset_config.preprocessing),
                "round": config.get("server_round", -1)
            }
            
        except Exception as e:
            print(f'[DEBUG] evaluate - EXCEPTION: {e}', flush=True)
            logger.error(f"Error during evaluation: {e}")
            raise

def ssl_client_fn(client_config: ClientConfig,
                  optimizer_type: str = "Adam",
                  ham10000_manager: Optional[HAM10000DataManager] = None):
    
    def client_fn(context: Context):
        
        adjusted_config = ClientConfig(
            client_id=client_config.client_id,
            ssl_task=client_config.ssl_task,
            dataset_config=client_config.dataset_config,
            optimizer=client_config.optimizer,
            learning_rate=client_config.learning_rate,
            batch_size=client_config.batch_size
        )
        
        return SSLClient(
            client_config=adjusted_config,
            optimizer_type=optimizer_type,
            ham10000_manager=ham10000_manager
        )
    
    return client_fn


def legacy_ssl_client_fn(partition_id: int, num_partitions: int, 
                        ssl_task: str = "rotation", 
                        optimizer_type: str = "Adam"):
    
    from .experimental_config import DatasetConfig, ClientConfig
    
    dataset_config = DatasetConfig(
        name="HAM10000",
        labeled_ratio=0.7,
        unlabeled_ratio=0.3,
        preprocessing=["resize"]
    )
    
    client_config = ClientConfig(
        client_id=partition_id + 1,
        ssl_task=ssl_task,
        dataset_config=dataset_config,
        optimizer=optimizer_type
    )
    
    def client_fn(context: Context):
        ham10000_manager = HAM10000DataManager(use_kagglehub=True, lazy_load=False)
        
        return SSLClient(
            client_config=client_config,
            optimizer_type=optimizer_type,
            ham10000_manager=ham10000_manager
        )
    
    return client_fn


def create_distributed_client(client_id: int, ssl_task: str, dataset_name: str, 
                            optimizer_type: str = "Adam") -> SSLClient:
    from .experimental_config import ExperimentalConfigurator
    
    client_configs = ExperimentalConfigurator.create_heterogeneous_clients()
    
    client_config = None
    for config in client_configs:
        if config.client_id == client_id:
            client_config = config
            break
    
    if client_config is None:
        if dataset_name == "HAM10000":
            dataset_config = DatasetConfig(
                name="HAM10000",
                labeled_ratio=0.7,
                unlabeled_ratio=0.3,
                preprocessing=["resize", "artifact_removal"]
            )
        else:
            raise ValueError(f"Unsupported dataset for distributed client: {dataset_name}")
        
        client_config = ClientConfig(
            client_id=client_id,
            ssl_task=ssl_task,
            dataset_config=dataset_config,
            optimizer=optimizer_type
        )

    ham10000_manager = HAM10000DataManager(use_kagglehub=True, lazy_load=False)
    
    return SSLClient(
        client_config=client_config,
        optimizer_type=optimizer_type,
        ham10000_manager=ham10000_manager
    )