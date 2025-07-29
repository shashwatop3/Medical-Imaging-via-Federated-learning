import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict
import itertools


class FrozenResNet50Backbone(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.feature_dim = 2048
    
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return features.view(features.size(0), -1)


class ContrastiveSSLModel(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = FrozenResNet50Backbone(pretrained)
        
        self.shared_head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.contrastive_head = nn.Linear(128, 128)
        
        self.temperature = 0.1
    
    def forward(self, x):
        features = self.backbone(x)
        shared_features = self.shared_head(features)
        projections = self.contrastive_head(shared_features)
        return F.normalize(projections, dim=1)
    
    def contrastive_loss(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        z = torch.cat([z_i, z_j], dim=0)
        
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        positive_pairs = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z.device)
        for i in range(batch_size):
            positive_pairs[i, i + batch_size] = True
            positive_pairs[i + batch_size, i] = True
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        loss = 0.0
        for i in range(2 * batch_size):
            if torch.any(positive_pairs[i]):
                pos_logits = sim_matrix[i][positive_pairs[i]]
                all_logits = sim_matrix[i][~mask[i]]
                
                max_logit = torch.max(all_logits)
                denominator = torch.log(torch.sum(torch.exp(all_logits - max_logit))) + max_logit
                
                for pos_logit in pos_logits:
                    loss += -(pos_logit - denominator)
        
        loss = loss / (2 * batch_size)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid contrastive loss detected. z_i norm: {torch.norm(z_i).item()}, z_j norm: {torch.norm(z_j).item()}")
            return torch.tensor(1.0, device=z.device, requires_grad=True)
        
        return loss
    
    def get_ssl_parameters(self):
        return [val.cpu().numpy() for name, val in self.shared_head.state_dict().items()]
    
    def set_ssl_parameters(self, parameters):
        param_names = list(self.shared_head.state_dict().keys())
        params_dict = zip(param_names, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.shared_head.load_state_dict(state_dict, strict=True)


class RotationSSLModel(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = FrozenResNet50Backbone(pretrained)
        
        self.shared_head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        shared_features = self.shared_head(features)
        rotation_logits = self.rotation_head(shared_features)
        return rotation_logits
    
    def get_ssl_parameters(self):
        return [val.cpu().numpy() for name, val in self.shared_head.state_dict().items()]
    
    def set_ssl_parameters(self, parameters):
        param_names = list(self.shared_head.state_dict().keys())
        params_dict = zip(param_names, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.shared_head.load_state_dict(state_dict, strict=True)


class JigsawSSLModel(nn.Module):
    
    def __init__(self, pretrained=True, num_patches=9):
        super().__init__()
        self.backbone = FrozenResNet50Backbone(pretrained)
        self.num_patches = num_patches
        
        self.jigsaw_head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim * num_patches, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1000)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        patch_features = []
        for i in range(self.num_patches):
            features = self.backbone(x[:, i])
            patch_features.append(features)
        
        combined_features = torch.cat(patch_features, dim=1)
        
        jigsaw_logits = self.jigsaw_head(combined_features)
        return jigsaw_logits
    
    def get_ssl_parameters(self):
        return [val.cpu().numpy() for name, val in self.jigsaw_head.state_dict().items()]
    
    def set_ssl_parameters(self, parameters):
        param_names = list(self.jigsaw_head.state_dict().keys())
        params_dict = zip(param_names, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.jigsaw_head.load_state_dict(state_dict, strict=False)


class DownstreamClassificationModel(nn.Module):
    
    def __init__(self, ssl_model, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = ssl_model.backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        if hasattr(self.backbone, 'backbone'):
            features = self.backbone(x)
        else:
            with torch.no_grad():
                features = self.backbone(x)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits


def create_ssl_model(task_type, **kwargs):
    if task_type == "rotation":
        return RotationSSLModel(**kwargs)
    elif task_type == "contrastive":
        return ContrastiveSSLModel(**kwargs)
    elif task_type == "jigsaw":
        return JigsawSSLModel(**kwargs)
    else:
        raise ValueError(f"Unknown SSL task type: {task_type}")


class NTXentLoss(nn.Module):
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, projections_1, projections_2):
        batch_size = projections_1.shape[0]
        
        projections = torch.cat([projections_1, projections_2], dim=0)
        
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        labels = labels - torch.eye(2 * batch_size)
        
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim * (1 - torch.eye(2 * batch_size))
        
        pos_sim = exp_sim * labels
        neg_sim = exp_sim.sum(dim=1, keepdim=True)
        
        loss = -torch.log(pos_sim.sum(dim=1) / neg_sim.squeeze())
        return loss.mean()