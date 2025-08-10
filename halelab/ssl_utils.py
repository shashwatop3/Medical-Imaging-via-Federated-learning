import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np
import random
from PIL import Image
import itertools
from pathlib import Path



class RotationDataset(Dataset):
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.base_transform = transform
        self.rotation_angles = [0, 90, 180, 270]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        if self.base_transform:
            image = self.base_transform(image)
        
        rotation_idx = random.randint(0, 3)
        angle = self.rotation_angles[rotation_idx]
        
        if isinstance(image, torch.Tensor):
            image_pil = TF.to_pil_image(image)
            rotated_image = TF.rotate(image_pil, angle)
            rotated_image = TF.to_tensor(rotated_image)
        else:
            rotated_image = TF.rotate(image, angle)
            rotated_image = TF.to_tensor(rotated_image)
        
        return rotated_image, rotation_idx


class ContrastiveDataset(Dataset):
    
    def __init__(self, image_paths, transform=None, strong_aug_prob=0.8):
        self.image_paths = image_paths
        self.base_transform = transform
        self.strong_aug_prob = strong_aug_prob
        
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.2, hue=0.05)
            ], p=0.6),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 1.0))
            ], p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        if self.base_transform:
            image = self.base_transform(image)
        
        view1 = self.augmentation(image)
        view2 = self.augmentation(image)
        
        if not isinstance(view1, torch.Tensor):
            view1 = TF.to_tensor(view1)
        if not isinstance(view2, torch.Tensor):
            view2 = TF.to_tensor(view2)
        
        return (view1, view2), 0


class JigsawDataset(Dataset):
    
    def __init__(self, image_paths, transform=None, patch_size=75, grid_size=3):
        self.image_paths = image_paths
        self.base_transform = transform
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        
        all_permutations = list(itertools.permutations(range(self.num_patches)))
        self.permutations = all_permutations[:100]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        if self.base_transform:
            image = self.base_transform(image)
        
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
        
        image_size = image.shape[1]
        actual_patch_size = image_size // self.grid_size
        
        effective_size = actual_patch_size * self.grid_size
        
        if image_size > effective_size:
            start = (image_size - effective_size) // 2
            image = image[:, start:start+effective_size, start:start+effective_size]
        
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                start_i = i * actual_patch_size
                end_i = start_i + actual_patch_size
                start_j = j * actual_patch_size
                end_j = start_j + actual_patch_size
                
                patch = image[:, start_i:end_i, start_j:end_j]
                patches.append(patch)
        
        perm_idx = random.randint(0, len(self.permutations) - 1)
        permutation = self.permutations[perm_idx]
        
        shuffled_patches = [patches[i] for i in permutation]
        
        rows = []
        for i in range(self.grid_size):
            row_patches = shuffled_patches[i * self.grid_size:(i + 1) * self.grid_size]
            row = torch.cat(row_patches, dim=2)
            rows.append(row)
        
        shuffled_image = torch.cat(rows, dim=1)
        
        return shuffled_image, perm_idx


class MedicalImageDataset(Dataset):
    
    def __init__(self, image_paths, labels=None, transform=None, ssl_task=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.ssl_task = ssl_task
        
        if ssl_task == "rotation":
            self.ssl_dataset = RotationDataset(image_paths, transform)
        elif ssl_task == "contrastive":
            self.ssl_dataset = ContrastiveDataset(image_paths, transform)
        elif ssl_task == "jigsaw":
            self.ssl_dataset = JigsawDataset(image_paths, transform)
        else:
            self.ssl_dataset = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.ssl_dataset:
            return self.ssl_dataset[idx]
        else:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            if self.labels is not None:
                return image, self.labels[idx]
            else:
                return image


def get_medical_image_paths(data_dir, extensions=['.png', '.jpg', '.jpeg']):
    data_path = Path(data_dir)
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(list(data_path.glob(f"**/*{ext}")))
    
    return [str(path) for path in image_paths]


def create_ssl_dataloader(image_paths, task_type, batch_size=32, shuffle=True, 
                         num_workers=4, **kwargs):
    
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if task_type == "rotation":
        dataset = RotationDataset(image_paths, base_transform)
    elif task_type == "contrastive":
        dataset = ContrastiveDataset(image_paths, base_transform)
    
    else:
        raise ValueError(f"Unknown SSL task type: {task_type}")
    
    pin_memory = torch.cuda.is_available() and not torch.backends.mps.is_available()
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     num_workers=num_workers, pin_memory=pin_memory)


def create_downstream_dataloader(image_paths, labels, batch_size=32, shuffle=True,
                               num_workers=4, augment=False):
    
    if augment:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    dataset = MedicalImageDataset(image_paths, labels, transform)
    
    pin_memory = torch.cuda.is_available() and not torch.backends.mps.is_available()
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                     num_workers=num_workers, pin_memory=pin_memory)


def partition_data(image_paths, labels, num_clients, partition_type="iid"):
    
    if partition_type == "iid":
        indices = np.random.permutation(len(image_paths))
        split_indices = np.array_split(indices, num_clients)
        
        client_data = []
        for client_indices in split_indices:
            client_paths = [image_paths[i] for i in client_indices]
            client_labels = [labels[i] for i in client_indices] if labels else None
            client_data.append((client_paths, client_labels))
        
        return client_data
    
    elif partition_type == "non_iid":
        if labels is None:
            raise ValueError("Labels required for non-IID partitioning")
        
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        client_data = [[] for _ in range(num_clients)]
        for class_label, indices in class_indices.items():
            np.random.shuffle(indices)
            split_indices = np.array_split(indices, num_clients)
            for client_id, client_indices in enumerate(split_indices):
                client_data[client_id].extend(client_indices)
        
        partitioned_data = []
        for client_indices in client_data:
            client_paths = [image_paths[i] for i in client_indices]
            client_labels = [labels[i] for i in client_indices]
            partitioned_data.append((client_paths, client_labels))
        
        return partitioned_data
    
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")