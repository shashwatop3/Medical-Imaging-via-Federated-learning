import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    KAGGLEHUB_AVAILABLE = True
    logger.info("KaggleHub is available for dataset loading")
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    logger.warning("KaggleHub not available. Install with: pip install kagglehub[pandas-datasets]")


class HAM10000Dataset(Dataset):
    
    def __init__(self, image_paths, labels=None, transform=None, ssl_task=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.ssl_task = ssl_task
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.ssl_task == 'rotation':
                angles = [0, 90, 180, 270]
                angle = np.random.choice(angles)
                image = image.rotate(angle)
                rot_label = angles.index(angle)
                
                image = self.transform(image)
                return image, rot_label
                
            elif self.ssl_task == 'contrastive':
                augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                view1 = augment_transform(image)
                view2 = augment_transform(image)
                return (view1, view2), 0
                
            elif self.ssl_task == 'jigsaw':
                image = self.transform(image)
                perm_idx = np.random.randint(0, 24)
                return image, perm_idx
            
            else:
                image = self.transform(image)
                
                if self.labels is not None:
                    label = self.labels[idx]
                    return image, label
                else:
                    return image, 0
                    
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, 0


class HAM10000DataManager:
    
    _cached_dataset_path = None
    _cached_metadata_df = None
    
    def __init__(self, use_kagglehub=True, lazy_load=False, data_dir=None):
        self.use_kagglehub = use_kagglehub
        self.data_dir = data_dir
        self.full_dataset = None
        self.metadata_df = None
        self.image_paths = []
        self.labels = []
        self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        if not lazy_load:
            self._load_dataset()
    
    def _load_dataset(self) -> bool:
        if self.use_kagglehub:
            return self._load_dataset_kagglehub()
        else:
            return self._load_dataset_local()
    
    def _load_dataset_kagglehub(self) -> bool:
        if not KAGGLEHUB_AVAILABLE:
            logger.warning("KaggleHub not available")
            return False
            
        try:
            if HAM10000DataManager._cached_metadata_df is not None and HAM10000DataManager._cached_dataset_path is not None:
                logger.info("Using cached dataset - no download needed!")
                self.metadata_df = HAM10000DataManager._cached_metadata_df.copy()
                dataset_path = Path(HAM10000DataManager._cached_dataset_path)
                
                image_dir = dataset_path / "images"
                if image_dir.exists():
                    self._setup_image_paths_fast(image_dir)
                    return True
            
            logger.info("Loading HAM10000 dataset using KaggleHub (first time - will cache for future use)...")
            
            try:
                logger.info("Loading metadata CSV using KaggleDatasetAdapter...")
                metadata_df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    "surajghuwalewala/ham1000-segmentation-and-classification",
                    "GroundTruth.csv"
                )
                logger.info(f"✓ Loaded metadata with {len(metadata_df)} records")
                
                HAM10000DataManager._cached_metadata_df = metadata_df.copy()
                
            except Exception as e:
                logger.warning(f"Could not load metadata via PANDAS adapter: {e}")
                logger.info("Downloading full dataset...")
                
                dataset_path = kagglehub.dataset_download("surajghuwalewala/ham1000-segmentation-and-classification")
                logger.info(f"✓ Dataset downloaded to: {dataset_path}")
                
                HAM10000DataManager._cached_dataset_path = str(dataset_path)
                
                dataset_path = Path(dataset_path)
                
                csv_files = list(dataset_path.glob("*.csv"))
                metadata_file = None
                
                for csv_file in csv_files:
                    if any(keyword in csv_file.name.lower() for keyword in ['groundtruth', 'metadata', 'labels']):
                        metadata_file = csv_file
                        break
                
                if metadata_file is None and csv_files:
                    metadata_file = csv_files[0]
                
                if metadata_file is None:
                    logger.error("No CSV metadata file found")
                    return False
                
                metadata_df = pd.read_csv(metadata_file)
                logger.info(f"✓ Loaded metadata from {metadata_file.name} with {len(metadata_df)} records")
                
                HAM10000DataManager._cached_metadata_df = metadata_df.copy()
                
                image_dir = dataset_path / "images"
                if not image_dir.exists():
                    for subdir in dataset_path.iterdir():
                        if subdir.is_dir() and any(subdir.glob("*.jpg")):
                            image_dir = subdir
                            break
                    else:
                        logger.error("No images directory found")
                        return False
            
            if 'dataset_path' not in locals():
                logger.info("Downloading full dataset for image files...")
                dataset_path = Path(kagglehub.dataset_download("surajghuwalewala/ham1000-segmentation-and-classification"))
                image_dir = dataset_path / "images"
                if not image_dir.exists():
                    for subdir in dataset_path.iterdir():
                        if subdir.is_dir() and any(subdir.glob("*.jpg")):
                            image_dir = subdir
                            break
                    else:
                        logger.error("No images directory found after dataset download")
                        return False
            
            self.metadata_df = metadata_df
            self.image_paths = []
            self.labels = []
            
            logger.info(f"Metadata columns: {list(metadata_df.columns)}")
            
            if 'image' in metadata_df.columns and any(col.upper() for col in metadata_df.columns if col.upper() in [c.upper() for c in self.class_names]):
                logger.info("Detected one-hot encoded label format")
                image_col = 'image'
                
                col_to_class = {}
                for col in metadata_df.columns:
                    for class_name in self.class_names:
                        if col.upper() == class_name.upper():
                            col_to_class[col] = class_name
                            break
                
                logger.info(f"Found class columns: {col_to_class}")
                
                found_images = 0
                for _, row in metadata_df.iterrows():
                    image_id = str(row[image_col])
                    
                    possible_names = [
                        f"{image_id}.jpg", f"{image_id}.JPG", 
                        f"{image_id}.png", f"{image_id}.PNG",
                        f"{image_id}.jpeg", f"{image_id}.JPEG"
                    ]
                    
                    image_path = None
                    for name in possible_names:
                        candidate_path = image_dir / name
                        if candidate_path.exists():
                            image_path = candidate_path
                            break
                    
                    if image_path and image_path.exists():
                        label_idx = None
                        for col, class_name in col_to_class.items():
                            if col in row and row[col] == 1:
                                label_idx = self.class_to_idx[class_name]
                                break
                        
                        if label_idx is not None:
                            self.image_paths.append(str(image_path))
                            self.labels.append(label_idx)
                            found_images += 1
                        else:
                            logger.warning(f"No valid label found for image {image_id}")
                    else:
                        logger.warning(f"Image not found for ID: {image_id}")
                        
            else:
                logger.info("Using traditional metadata format")
                
                possible_id_cols = ['image_id', 'image', 'filename', 'id', 'lesion_id']
                possible_dx_cols = ['dx', 'diagnosis', 'class', 'label', 'category']
                
                id_col = None
                dx_col = None
                
                for col in metadata_df.columns:
                    col_lower = col.lower()
                    if id_col is None and any(keyword in col_lower for keyword in possible_id_cols):
                        id_col = col
                    if dx_col is None and any(keyword in col_lower for keyword in possible_dx_cols):
                        dx_col = col
                
                if id_col is None:
                    id_col = metadata_df.columns[0]
                    logger.warning(f"No clear image ID column found, using: {id_col}")
                
                if dx_col is None:
                    logger.warning("No diagnosis column found, creating dummy labels")
                    metadata_df['dx'] = ['nv'] * len(metadata_df)
                    dx_col = 'dx'
                
                logger.info(f"Using ID column: {id_col}, Diagnosis column: {dx_col}")
                
                found_images = 0
                for _, row in metadata_df.iterrows():
                    image_id = str(row[id_col])
                    
                    possible_names = [
                        f"{image_id}.jpg", f"{image_id}.JPG", 
                        f"{image_id}.png", f"{image_id}.PNG",
                        f"{image_id}.jpeg", f"{image_id}.JPEG"
                    ]
                    
                    image_path = None
                    for name in possible_names:
                        candidate_path = image_dir / name
                        if candidate_path.exists():
                            image_path = candidate_path
                            break
                    
                    if image_path and image_path.exists():
                        diagnosis = str(row[dx_col]).lower()
                        if diagnosis in self.class_to_idx:
                            label = self.class_to_idx[diagnosis]
                            self.image_paths.append(str(image_path))
                            self.labels.append(label)
                            found_images += 1
                        else:
                            logger.warning(f"Unknown diagnosis: {diagnosis} for image {image_id}")
                    else:
                        logger.warning(f"Image not found for ID: {image_id}")
            
            logger.info(f"Successfully loaded {found_images} images with labels")
            
            if found_images == 0:
                logger.error("No images found with valid labels")
                return False
            
            logger.info(f"Class distribution: {dict(zip(self.class_names, np.bincount(self.labels)))}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset via KaggleHub: {e}")
            return False
    
    def _load_dataset_local(self) -> bool:
        if not self.data_dir or not self.data_dir.exists():
            logger.warning(f"Local data directory not found: {self.data_dir}")
            return False
        
        try:
            metadata_files = [
                self.data_dir / "HAM10000_metadata.csv",
                self.data_dir / "metadata.csv",
                self.data_dir / "hmnist_28_28_RGB.csv"
            ]
            
            metadata_file = None
            for file in metadata_files:
                if file.exists():
                    metadata_file = file
                    break
            
            if metadata_file is None:
                logger.error("No metadata file found in local directory")
                return False
            
            self.metadata_df = pd.read_csv(metadata_file)
            logger.info(f"Loaded local metadata with {len(self.metadata_df)} records")
            
            image_dirs = [
                self.data_dir / "images",
                self.data_dir / "HAM10000_images",
                self.data_dir
            ]
            
            image_dir = None
            for img_dir in image_dirs:
                if img_dir.exists() and any(img_dir.glob("*.jpg")):
                    image_dir = img_dir
                    break
            
            if image_dir is None:
                logger.error("No images found in local directory")
                return False
            
            self.image_paths = []
            self.labels = []
            
            for _, row in self.metadata_df.iterrows():
                image_id = row.get('image_id', row.get('lesion_id', row.iloc[0]))
                image_path = image_dir / f"{image_id}.jpg"
                
                if image_path.exists():
                    self.image_paths.append(str(image_path))
                    label_name = row.get('dx', row.get('diagnosis', 'nv'))
                    label_idx = self.class_to_idx.get(label_name, self.class_to_idx['nv'])
                    self.labels.append(label_idx)
            
            logger.info(f"Successfully loaded {len(self.image_paths)} local images")
            return len(self.image_paths) > 0
            
        except Exception as e:
            logger.error(f"Error loading local dataset: {e}")
            return False
    
    def _setup_image_paths_fast(self, image_dir: Path):
        if self.metadata_df is None:
            return False
            
        logger.info("Setting up image paths (fast mode)...")
        self.image_paths = []
        self.labels = []
        
        if 'image' in self.metadata_df.columns:
            image_col = 'image'
            
            col_to_class = {}
            for col in self.metadata_df.columns:
                for class_name in self.class_names:
                    if col.upper() == class_name.upper():
                        col_to_class[col] = class_name
                        break
            
            found_images = 0
            total_records = len(self.metadata_df)
            
            for idx, (_, row) in enumerate(self.metadata_df.iterrows()):
                if idx % 1000 == 0:
                    logger.info(f"Processing records: {idx}/{total_records}")
                    
                image_id = str(row[image_col])
                
                for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
                    image_path = image_dir / f"{image_id}{ext}"
                    if image_path.exists():
                        for col, class_name in col_to_class.items():
                            if col in row and row[col] == 1:
                                self.image_paths.append(str(image_path))
                                self.labels.append(self.class_to_idx[class_name])
                                found_images += 1
                                break
                        break
            
            logger.info(f"✓ Fast setup complete: {found_images} images indexed")
            return found_images > 0
        
        return False
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        if len(self.image_paths) == 0:
            return [], [], [], []
        
        from sklearn.model_selection import train_test_split
        
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.image_paths, self.labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.labels
        )
        
        return train_paths, test_paths, train_labels, test_labels
    
    def get_federated_splits(self, num_clients: int = 3, method: str = 'iid'):
        if len(self.image_paths) == 0:
            return [[] for _ in range(num_clients)]
        
        if method == 'iid':
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            splits = np.array_split(indices, num_clients)
            
            client_data = []
            for split in splits:
                client_paths = [self.image_paths[i] for i in split]
                client_labels = [self.labels[i] for i in split]
                client_data.append((client_paths, client_labels))
            
            return client_data
        
        else:
            class_indices = {i: [] for i in range(len(self.class_names))}
            
            for idx, label in enumerate(self.labels):
                class_indices[label].append(idx)
            
            client_data = [[] for _ in range(num_clients)]
            
            for class_idx, indices in class_indices.items():
                np.random.shuffle(indices)
                client_splits = np.array_split(indices, num_clients)
                
                for client_id, split in enumerate(client_splits):
                    for idx in split:
                        client_data[client_id].append(idx)
            
            result = []
            for client_indices in client_data:
                client_paths = [self.image_paths[i] for i in client_indices]
                client_labels = [self.labels[i] for i in client_indices]
                result.append((client_paths, client_labels))
            
            return result
        
    def create_federated_splits(self, num_clients: int = 3, split_type: str = 'iid'):
        return self.get_federated_splits(num_clients, method=split_type)
    
    def create_ssl_dataset(self, image_paths: List[str], task_type: str = "rotation", augment: bool = True):
        from torchvision import transforms
        
        if augment:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        ssl_dataset = HAM10000Dataset(
            image_paths=image_paths, 
            labels=None,
            transform=transform,
            ssl_task=task_type
        )
        
        return ssl_dataset

    def create_heterogeneous_client_partitions(self, num_clients: int = 3) -> Dict[int, Tuple[List[str], List[int]]]:
        if len(self.image_paths) == 0:
            logger.warning("No data available for partitioning")
            return {}
        
        total_samples = len(self.image_paths)
        partition_size = total_samples // num_clients
        
        client_partitions = {}
        
        for client_id in range(num_clients):
            start_idx = client_id * partition_size
            if client_id == num_clients - 1:
                end_idx = total_samples
            else:
                end_idx = start_idx + partition_size
            
            client_paths = self.image_paths[start_idx:end_idx]
            client_labels = self.labels[start_idx:end_idx]
            
            client_partitions[client_id] = (client_paths, client_labels)
            
            logger.info(f"Client {client_id} partition: {len(client_paths)} samples")
        
        return client_partitions

    def create_dataloader(self, client_id: int, num_clients: int, ssl_task: str,
                            labeled_ratio: float = 1.0, batch_size: int = 16,
                            shuffle: bool = True, preprocessing: List[str] = None) -> torch.utils.data.DataLoader:
        if client_id < 0 or client_id >= num_clients:
            raise ValueError(f"client_id {client_id} must be between 0 and {num_clients-1}")
        
        client_partitions = self.create_heterogeneous_client_partitions(num_clients)
        
        if client_id not in client_partitions:
            raise ValueError(f"No partition found for client_id {client_id}")
        
        client_paths, client_labels = client_partitions[client_id]
        
        if len(client_paths) == 0:
            logger.warning(f"Client {client_id} has no data. Checking if any data available...")
            if len(self.image_paths) == 0:
                logger.error("No data available in the dataset")
                raise ValueError("Dataset is empty - no images available")
            
            logger.warning(f"Creating dummy dataset for client {client_id} with single sample")
            client_paths = [self.image_paths[0]]
            client_labels = [self.labels[0] if self.labels else 0]
        
        if labeled_ratio < 1.0:
            num_labeled = max(1, int(len(client_paths) * labeled_ratio))
            client_paths = client_paths[:num_labeled]
            client_labels = client_labels[:num_labeled]
        
        transform = self._create_preprocessing_transform(preprocessing or [])
        
        dataset = HAM10000Dataset(
            image_paths=client_paths,
            labels=client_labels,
            transform=transform,
            ssl_task=ssl_task
        )
        
        if len(dataset) == 0:
            logger.error(f"Dataset for client {client_id} is empty after processing")
            raise ValueError(f"Client {client_id} dataset is empty")
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Created dataloader for client {client_id} with {len(dataset)} samples")
        return dataloader

    def _create_preprocessing_transform(self, preprocessing: List[str]) -> transforms.Compose:
        from torchvision import transforms
        
        transform_list = [transforms.Resize((224, 224))]
        
        if "rotation" in preprocessing:
            transform_list.append(transforms.RandomRotation(degrees=30))
        
        if "color_jitter" in preprocessing:
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        
        if "flip" in preprocessing:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if "gaussian_blur" in preprocessing:
            transform_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))
        
        if "blur" in preprocessing:
            transform_list.append(transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)

    def create_downstream_dataset(self, image_paths: List[str], labels: List[int]) -> HAM10000Dataset:
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        return HAM10000Dataset(
            image_paths=image_paths,
            labels=labels,
            transform=transform,
            ssl_task=None
        )

    def get_stats(self) -> Dict[str, Any]:
        if len(self.image_paths) == 0:
            return {"error": "No data loaded"}
        
        stats = {
            "total_samples": len(self.image_paths),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "class_distribution": {}
        }
        
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            count = self.labels.count(class_idx)
            stats["class_distribution"][class_name] = count
        
        return stats
    
    def get_subset(self, n=10):
        if hasattr(self, 'full_dataset') and self.full_dataset is not None:
            return list(self.full_dataset)[:n]
        if hasattr(self, 'metadata_df') and self.metadata_df is not None:
            return list(self.metadata_df.index[:n])
        return list(range(n))


def create_ham10000_ssl_dataloader(data_manager, partition_id, num_partitions, ssl_task, batch_size):
    import torch
    from torch.utils.data import DataLoader
    
    def contrastive_collate_fn(batch):
        if ssl_task == 'contrastive':
            views = []
            ssl_labels = []
            batch_indices = []
            
            for i, ((view1, view2), ssl_label) in enumerate(batch):
                views.append((view1, view2))
                ssl_labels.append(ssl_label)
                batch_indices.append(i)
            
            view1_batch = torch.stack([v[0] for v in views])
            view2_batch = torch.stack([v[1] for v in views])
            ssl_labels_batch = torch.tensor(ssl_labels)
            
            return (view1_batch, view2_batch), ssl_labels_batch
        else:
            return torch.utils.data.default_collate(batch)
    
    if not data_manager.image_paths:
        data_manager._load_dataset()
    
    client_partitions = data_manager.create_heterogeneous_client_partitions(num_partitions)
    
    if partition_id not in client_partitions:
        raise ValueError(f"Partition {partition_id} not found in available partitions")
    
    image_paths, labels = client_partitions[partition_id]
    
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    if labels is not None:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Stratified split: train={len(train_paths)}, val={len(val_paths)}")
        print(f"Train class distribution: {np.bincount(train_labels)}")
        print(f"Val class distribution: {np.bincount(val_labels)}")
    else:
        split_idx = int(len(image_paths) * 0.8)
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]
        train_labels = None
        val_labels = None
    
    train_dataset = data_manager.create_ssl_dataset(train_paths, task_type=ssl_task, augment=True)
    
    if ssl_task == 'contrastive':
        val_dataset = data_manager.create_ssl_dataset(val_paths, task_type=ssl_task, augment=False)
        val_dataset.downstream_labels = val_labels
    else:
        val_dataset = data_manager.create_ssl_dataset(val_paths, task_type=ssl_task, augment=False)
        val_dataset.downstream_labels = val_labels
    
    collate_fn = contrastive_collate_fn if ssl_task == 'contrastive' else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    print(f"Created SSL dataloaders for client {partition_id}: train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"Using {'custom contrastive' if ssl_task == 'contrastive' else 'default'} collate function")
    return train_loader, val_loader


def create_ham10000_downstream_dataloader(data_manager: HAM10000DataManager,
                                         image_paths: List[str],
                                         labels: List[int],
                                         batch_size: int = 32,
                                         shuffle: bool = True) -> DataLoader:
    
    dataset = HAM10000Dataset(image_paths, labels=labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    return dataloader


def download_ham10000_sample(output_dir: str = "/Users/Codes/HaleLab/HAM10000_sample") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if KAGGLEHUB_AVAILABLE:
        try:
            logger.info("Downloading HAM10000 sample using KaggleHub...")
            
            dataset_path = kagglehub.dataset_download("surajghuwalewala/ham1000-segmentation-and-classification")
            
            import shutil
            from random import sample
            
            source_path = Path(dataset_path)
            
            jpg_files = list(source_path.rglob("*.jpg"))
            
            if len(jpg_files) > 100:
                sample_files = sample(jpg_files, 100)
            else:
                sample_files = jpg_files
            
            sample_image_dir = output_path / "images"
            sample_image_dir.mkdir(exist_ok=True)
            
            for img_file in sample_files:
                shutil.copy2(img_file, sample_image_dir / img_file.name)
            
            metadata_files = list(source_path.rglob("*metadata*.csv"))
            if metadata_files:
                shutil.copy2(metadata_files[0], output_path / "HAM10000_metadata.csv")
            
            logger.info(f"Sample dataset created at {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading sample: {e}")
    
    logger.info("Creating minimal sample structure for testing...")
    
    sample_image_dir = output_path / "images"
    sample_image_dir.mkdir(exist_ok=True)
    
    metadata = pd.DataFrame({
        'image_id': [f'sample_{i:03d}' for i in range(10)],
        'dx': ['nv', 'mel', 'bcc', 'akiec', 'bkl', 'df', 'vasc'] * 2,
        'dx_type': ['histo'] * 10,
        'age': [30, 45, 60, 35, 50, 40, 55, 25, 65, 38],
        'sex': ['male', 'female'] * 5,
        'localization': ['back', 'face', 'chest', 'hand', 'leg'] * 2
    })
    
    metadata.to_csv(output_path / "HAM10000_metadata.csv", index=False)
    
    logger.info(f"Minimal sample created at {output_path}")
    logger.info("Note: This is a test structure. Download the real HAM10000 dataset for actual use.")
    
    return str(output_path)