"""Fashion Network Model Implementation For Deep Fashion Dataset.

This module implements the FashionNet architecture with multi-task learning
for category classification, attribute prediction, and landmark detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torch.nn.functional as F

class LandmarkBranch(nn.Module):
    """Local Feature Branch For Landmark Prediction And Pooling.
    
    Args:
        in_channels (int): Number of input channels from backbone features
        num_landmarks (int): Number of landmarks to predict
    """
    def __init__(self, in_channels=512, num_landmarks=8):
        """Initialize Landmark Branch."""
        super().__init__()
        
        # landmark prediction layers with deeper network
        self.landmark_pred = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, num_landmarks * 3, kernel_size=1)  # x, y, visibility per landmark
        )
        
        # local feature extraction with batch norm
        self.local_features = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        """Forward Pass Through Landmark Branch.
        
        Args:
            x (torch.Tensor): Input features from backbone [batch_size, channels, h, w]
            
        Returns:
            tuple: (landmark_pred, local_features)
                - landmark_pred: [batch_size, num_landmarks * 3, h, w]
                - local_features: [batch_size, 512, h, w]
        """
        landmarks = self.landmark_pred(x)
        local_feat = self.local_features(x)
        return landmarks, local_feat

class AttributeGroup(nn.Module):
    """Tree-Structured Attribute Group Prediction Module.
    
    Args:
        in_features (int): Input feature dimension
        num_attributes (int): Number of attributes in this group
    """
    def __init__(self, in_features, num_attributes):
        """Initialize Attribute Group."""
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_attributes)
        )
        
    def forward(self, x):
        """Forward Pass Through Attribute Group.
        
        Args:
            x (torch.Tensor): Input features [batch_size, in_features]
            
        Returns:
            torch.Tensor: Attribute predictions [batch_size, num_attributes]
        """
        return self.fc(x)

class FashionNet(nn.Module):
    """Multi-Task Fashion Network For Category And Attribute Prediction.
    
    Args:
        num_categories (int): Number of clothing categories
        attr_group_dims (list): List of attribute counts for each group
        num_landmarks (int): Number of landmarks to predict
        pretrained (bool): Whether to use pretrained backbone
    """
    def __init__(self, num_categories=50, attr_group_dims=[7,3,3,4,6,3], num_landmarks=8, pretrained=True):
        """Initialize Fashion Network."""
        super().__init__()
        
        # load backbone with proper weights
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.vgg16(weights=weights)
        
        # feature extractor (up to last conv layer)
        self.features = backbone.features
        
        # global branch for category and attribute prediction
        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # category prediction branch with attention
        self.category_attention = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        self.category_branch = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_categories)
        )
        
        # attribute prediction branches
        self.attr_groups = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 * 7 * 7, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, dim)
            ) for dim in attr_group_dims
        ])
        
        # landmark prediction branch
        self.landmark_branch = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),  # global pooling
            nn.Conv2d(256, num_landmarks * 3, 1)  # predict x, y, visibility for each landmark
        )
        
        # feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512 * 7 * 7 + num_landmarks * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize Model Weights Using Kaiming/Normal Initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def extract_local_features(self, features, landmarks, visibility):
        """Extract Local Features Around Landmark Points.
        
        Args:
            features (torch.Tensor): Feature maps [batch_size, channels, h, w]
            landmarks (torch.Tensor): Landmark coordinates [batch_size, num_landmarks, 2]
            visibility (torch.Tensor): Landmark visibility [batch_size, num_landmarks]
            
        Returns:
            torch.Tensor: Local features [batch_size, channels * num_landmarks]
        """
        batch_size, channels, h, w = features.shape
        num_landmarks = landmarks.shape[1]
        
        # normalize landmark coordinates to [-1, 1] for grid_sample
        landmarks = landmarks.clone()
        landmarks[:, :, 0] = (landmarks[:, :, 0] / (w - 1)) * 2 - 1  # x coord
        landmarks[:, :, 1] = (landmarks[:, :, 1] / (h - 1)) * 2 - 1  # y coord
        
        # create sampling grid for 7x7 patches around each landmark
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-3, 3, 7),
            torch.linspace(-3, 3, 7),
            indexing='ij'
        )
        grid_offset = torch.stack([grid_x, grid_y], dim=-1).to(landmarks.device) / 14  # scale offsets
        
        # extract features around each landmark
        local_features = []
        for i in range(num_landmarks):
            # create grid centered at landmark with 7x7 offsets
            center = landmarks[:, i:i+1, :].view(batch_size, 1, 1, 2)
            grid = center + grid_offset.view(1, 7, 7, 2)
            grid = torch.clamp(grid, -1, 1)  # ensure valid sampling coordinates
            
            # sample features and apply visibility mask
            local_feat = torch.nn.functional.grid_sample(features, grid, align_corners=True)
            local_feat = local_feat * visibility[:, i:i+1, None, None]  # mask invisible landmarks
            local_feat = self.local_pool(local_feat)  # pool to 1x1
            local_features.append(local_feat.view(batch_size, -1))
            
        return torch.cat(local_features, dim=1)
    
    def forward(self, x):
        """Forward Pass Through The Network.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, 3, 224, 224]
            
        Returns:
            tuple: (category_pred, attr_preds, landmark_pred)
                - category_pred: Category logits [batch_size, num_categories]
                - attr_preds: List of attribute logits per group
                - landmark_pred: Landmark coordinates and visibility [batch_size, num_landmarks, 3]
        """
        # extract features from backbone
        features = self.features(x)
        
        # global branch processing
        global_feat = self.global_pool(features)
        
        # category prediction with attention
        attention = self.category_attention(global_feat)
        attended_feat = global_feat * attention
        category_feat = attended_feat.view(attended_feat.size(0), -1)
        category_pred = self.category_branch(category_feat)
        
        # attribute prediction for each group
        global_feat_flat = global_feat.view(global_feat.size(0), -1)
        attr_preds = [group(global_feat_flat) for group in self.attr_groups]
        
        # landmark prediction and normalization
        landmark_pred = self.landmark_branch(features)  # [batch_size, num_landmarks * 3, 1, 1]
        landmark_pred = landmark_pred.squeeze(-1).squeeze(-1)  # [batch_size, num_landmarks * 3]
        batch_size = landmark_pred.size(0)
        num_landmarks = 8
        landmark_pred = landmark_pred.view(batch_size, num_landmarks, 3)
        
        # normalize landmark coordinates to [0, 1] and visibility
        landmark_coords = torch.sigmoid(landmark_pred[:, :, :2])  # normalize coordinates to [0,1]
        landmark_vis = torch.sigmoid(landmark_pred[:, :, 2])  # normalize visibility to [0,1]
        
        # combine normalized predictions
        landmark_pred = torch.cat([
            landmark_coords,
            landmark_vis.unsqueeze(-1)
        ], dim=-1)
        
        return category_pred, attr_preds, landmark_pred