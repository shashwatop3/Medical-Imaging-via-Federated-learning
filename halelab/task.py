from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.metrics import precision_recall_fscore_support
import itertools

from .ssl_models import (
    ContrastiveSSLModel,
    RotationSSLModel, 
    JigsawSSLModel,
    DownstreamClassificationModel, 
    NTXentLoss
)
from .ssl_utils import (
    create_ssl_dataloader, 
    create_downstream_dataloader,
    get_medical_image_paths,
    partition_data
)
from .ham10000_utils import (
    HAM10000DataManager,
    create_ham10000_ssl_dataloader,
    create_ham10000_downstream_dataloader
)


def get_ssl_model(task_type, **kwargs):
    if task_type == "contrastive":
        return ContrastiveSSLModel(**kwargs)
    elif task_type == "rotation":
        return RotationSSLModel(**kwargs)
    elif task_type == "jigsaw":
        return JigsawSSLModel(**kwargs)
    else:
        raise ValueError(f"Unknown SSL task type: {task_type}")


def train_ssl(model, dataloader, epochs, device, task_type):
    model.to(device)
    model.train()
    
    if task_type == "rotation":
        optimizer = torch.optim.Adam(
            itertools.chain(model.shared_head.parameters(), model.rotation_head.parameters()), 
            lr=0.005,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
    elif task_type == "contrastive":
        optimizer = torch.optim.Adam(
            itertools.chain(model.shared_head.parameters(), model.contrastive_head.parameters()), 
            lr=0.001
        )
    elif task_type == "jigsaw":
        optimizer = torch.optim.Adam(model.jigsaw_head.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown SSL task type: {task_type}")
    
    print(f"Training {task_type} SSL model for {epochs} epochs")
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        max_batches = min(50, len(dataloader))
        
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            optimizer.zero_grad()
            
            if task_type == "rotation":
                images, rotation_labels = batch_data
                images, rotation_labels = images.to(device), rotation_labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, rotation_labels)
                
                if batch_idx % 20 == 0:
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == rotation_labels).sum().item()
                    accuracy = correct / rotation_labels.size(0)
                    print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
                    print(f"   Labels: {rotation_labels[:8].tolist()}, Predicted: {predicted[:8].tolist()}")
                
            elif task_type == "contrastive":
                views, _ = batch_data
                img1, img2 = views[0].to(device), views[1].to(device)
                
                z1 = model(img1)
                z2 = model(img2)
                loss = model.contrastive_loss(z1, z2)
                
            elif task_type == "jigsaw":
                patches, perm_labels = batch_data
                patches, perm_labels = patches.to(device), perm_labels.to(device)
                
                outputs = model(patches)
                loss = criterion(outputs, perm_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"✅ Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
    
    print(f"🎉 {task_type.capitalize()} SSL training completed!")
    
    return avg_loss





def evaluate_ssl(model, dataloader, device, task_type):
    model.to(device)
    model.eval()
    
    total_loss = 0
    total_metric = 0
    
    with torch.no_grad():
        for data in dataloader:
            if task_type == "contrastive":
                views, _ = data
                view1, view2 = views[0].to(device), views[1].to(device)
                
                z1 = model(view1)
                z2 = model(view2)
                
                loss = model.contrastive_loss(z1, z2)
                
                metric = loss.item()
                
            elif task_type == "rotation":
                images, rotation_labels = data
                images, rotation_labels = images.to(device), rotation_labels.to(device)
                
                outputs = model(images)
                
                loss = nn.CrossEntropyLoss()(outputs, rotation_labels)
                
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == rotation_labels).sum().item()
                metric = correct / rotation_labels.size(0)
                
            elif task_type == "jigsaw":
                shuffled_images, perm_labels = data
                shuffled_images, perm_labels = shuffled_images.to(device), perm_labels.to(device)
                
                outputs = model(shuffled_images)
                
                loss = nn.CrossEntropyLoss()(outputs, perm_labels)
                
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == perm_labels).sum().item()
                metric = correct / perm_labels.size(0)
                
            else:
                raise ValueError(f"Unknown SSL task type: {task_type}")
            
            total_loss += loss.item()
            total_metric += metric
    
    avg_loss = total_loss / len(dataloader)
    avg_metric = total_metric / len(dataloader)
    
    return avg_loss, avg_metric


def load_ssl_data(partition_id, num_partitions, task_type, use_kagglehub=True):
    try:
        data_manager = HAM10000DataManager(use_kagglehub=use_kagglehub, lazy_load=True)
        
        train_loader, val_loader = create_ham10000_ssl_dataloader(
            data_manager=data_manager,
            partition_id=partition_id,
            num_partitions=num_partitions,
            ssl_task=task_type,
            batch_size=8
        )
        
        print(f"Using HAM10000 dataset via {'KaggleHub' if use_kagglehub else 'local files'}")
        print(f"Client {partition_id}: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"Error loading HAM10000 dataset: {e}")
        raise RuntimeError(f"Failed to load HAM10000 dataset for federated training: {e}")


def get_ssl_weights(model):
    return model.get_ssl_parameters()


def set_ssl_weights(model, parameters):
    model.set_ssl_parameters(parameters)


def get_model(**kwargs):
    num_classes = kwargs.get('num_classes', 7)
    pretrained = kwargs.get('pretrained', True)
    
    return DownstreamClassificationModel(
        num_classes=num_classes,
        pretrained=pretrained
    )


def get_weights(model):
    if hasattr(model, 'get_ssl_parameters') and hasattr(model, 'set_ssl_parameters'):
        return model.get_ssl_parameters()
    else:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    if hasattr(model, 'get_ssl_parameters') and hasattr(model, 'set_ssl_parameters'):
        model.set_ssl_parameters(parameters)
    else:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)


def train_downstream(model, dataloader, epochs, device):
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
        
        print(f"Downstream Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    return running_loss / (epochs * len(dataloader))


def evaluate_downstream(model, dataloader, device):
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1


def load_ham10000_downstream_data(data_dir="/Users/Codes/HaleLab/HAM10000_dataset", test_size=0.3):
    try:
        data_manager = HAM10000DataManager(use_kagglehub=True, lazy_load=False)
        image_paths = data_manager.image_paths
        labels = data_manager.labels
        
        if not image_paths:
            raise RuntimeError("No images were loaded by the HAM10000DataManager.")
        
        from sklearn.model_selection import train_test_split
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42, 
            stratify=labels if len(set(labels)) > 1 else None
        )
        
        print(f"HAM10000 Downstream Data:")
        print(f"  Training samples: {len(train_paths)}")
        print(f"  Test samples: {len(test_paths)}")
        print(f"  Number of classes: {len(set(labels))}")
        print(f"  Class distribution: {np.bincount(labels)}")
        
        return train_paths, test_paths, train_labels, test_labels, data_manager
        
    except Exception as e:
        print(f"Error loading HAM10000 downstream data: {e}")
        return None, None, None, None, None


def create_ham10000_dataloaders(data_manager, train_paths, test_paths, 
                               train_labels, test_labels, batch_size=16):
    
    train_loader = create_ham10000_downstream_dataloader(
        data_manager=data_manager,
        image_paths=train_paths,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = create_ham10000_downstream_dataloader(
        data_manager=data_manager,
        image_paths=test_paths,
        labels=test_labels,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    
    print("Testing SSL data loading...")
    train_loader, val_loader = load_ssl_data(
        partition_id=0, 
        num_partitions=2, 
        task_type="rotation"
    )
    
    print("\nTesting SSL model creation...")
    ssl_model = get_ssl_model("rotation")
    
    print("\nTesting SSL model training...")
    train_ssl(
        model=ssl_model,
        dataloader=train_loader,
        epochs=1,
        device="cpu",
        task_type="rotation"
    )
    
    print("\nTesting SSL model evaluation...")
    ssl_loss, ssl_metric = evaluate_ssl(
        model=ssl_model,
        dataloader=val_loader,
        device="cpu",
        task_type="rotation"
    )
    print(f"SSL Evaluation - Loss: {ssl_loss:.4f}, Metric: {ssl_metric:.4f}")
    
    print("\nTesting downstream classification...")
    
    train_paths, test_paths, train_labels, test_labels, data_manager = load_ham10000_downstream_data()
    
    if data_manager:
        train_loader_downstream, test_loader_downstream = create_ham10000_dataloaders(
            data_manager, train_paths, test_paths, train_labels, test_labels
        )
        
        downstream_model = DownstreamClassificationModel(
            ssl_model=ssl_model,
            num_classes=7
        )
        
        print("\nTraining downstream model...")
        train_downstream(
            model=downstream_model,
            dataloader=train_loader_downstream,
            epochs=1,
            device="cpu"
        )
        
        print("\nEvaluating downstream model...")
        loss, acc, prec, rec, f1 = evaluate_downstream(
            model=downstream_model,
            dataloader=test_loader_downstream,
            device="cpu"
        )
        print(f"Downstream Evaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
    else:
        print("Skipping downstream tests due to data loading failure.")