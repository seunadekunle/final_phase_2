import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal loss for better handling of hard examples."""
    
    def __init__(self, gamma=2.0, alpha=None):
        """Initialize focal loss.
        
        Args:
            gamma (float): Focusing parameter
            alpha (torch.Tensor, optional): Class weights
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, input, target):
        """Compute focal loss.
        
        Args:
            input (torch.Tensor): Predicted logits [batch_size, num_classes]
            target (torch.Tensor): Target labels [batch_size]
            
        Returns:
            torch.Tensor: Loss value
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""
    
    def __init__(self, num_classes, smoothing=0.1):
        """Initialize label smoothing loss.
        
        Args:
            num_classes (int): Number of classes
            smoothing (float): Smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, input, target):
        """Compute label smoothing loss.
        
        Args:
            input (torch.Tensor): Predicted logits [batch_size, num_classes]
            target (torch.Tensor): Target labels [batch_size]
            
        Returns:
            torch.Tensor: Loss value
        """
        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()

class FashionNetLoss(nn.Module):
    """Loss function for FashionNet model."""
    
    def __init__(self, category_weight=1.0, attribute_weight=1.0, landmark_weight=1.0):
        super().__init__()
        self.category_weight = category_weight
        self.attribute_weight = attribute_weight
        self.landmark_weight = landmark_weight
        
        # category loss with label smoothing and focal loss
        self.category_criterion = LabelSmoothingLoss(num_classes=50, smoothing=0.1)
        self.focal_loss = FocalLoss(gamma=2.0)
        
        # attribute loss with positive/negative balancing
        self.pos_weight = torch.tensor([5.0])  # weight positive samples more
        
    def category_loss(self, pred, target):
        """Compute category classification loss.
        
        Args:
            pred (torch.Tensor): Predicted logits [batch_size, num_categories]
            target (torch.Tensor): Target labels [batch_size]
            
        Returns:
            torch.Tensor: Loss value
        """
        # combine label smoothing and focal loss
        smooth_loss = self.category_criterion(pred, target)
        focal_loss = self.focal_loss(pred, target)
        return 0.5 * (smooth_loss + focal_loss)
    
    def attribute_loss(self, pred, target):
        """Compute balanced binary cross entropy loss for attributes."""
        total_loss = 0
        start_idx = 0
        
        for group_pred in pred:
            num_attrs = group_pred.size(1)
            group_target = target[:, start_idx:start_idx + num_attrs]
            
            # compute positive sample weights for this group
            num_pos = torch.sum(group_target, dim=0)
            num_neg = group_target.size(0) - num_pos
            pos_weights = torch.where(num_pos > 0, num_neg.float() / num_pos.float(), self.pos_weight)
            pos_weights = pos_weights.to(pred[0].device)
            
            # compute balanced BCE loss
            loss = F.binary_cross_entropy_with_logits(
                group_pred,
                group_target,
                pos_weight=pos_weights,
                reduction='mean'
            )
            
            total_loss += loss
            start_idx += num_attrs
            
        return total_loss / len(pred)
    
    def landmark_loss(self, pred, target, visibility):
        """Compute landmark prediction loss."""
        # coordinate regression loss (only for visible landmarks)
        coord_loss = F.mse_loss(
            pred[:, :, :2] * visibility.unsqueeze(-1),
            target * visibility.unsqueeze(-1)
        )
        
        # visibility prediction loss
        vis_loss = F.binary_cross_entropy_with_logits(
            pred[:, :, 2],
            visibility
        )
        
        return coord_loss + 0.5 * vis_loss
    
    def forward(self, predictions, targets):
        """Compute total loss."""
        category_pred, attr_preds, landmark_pred = predictions
        categories, attributes, landmark_coords, landmark_vis = targets
        
        # compute individual losses
        cat_loss = self.category_loss(category_pred, categories)
        attr_loss = self.attribute_loss(attr_preds, attributes)
        lm_loss = self.landmark_loss(landmark_pred, landmark_coords, landmark_vis)
        
        # compute weighted total loss
        total_loss = (
            self.category_weight * cat_loss +
            self.attribute_weight * attr_loss +
            self.landmark_weight * lm_loss
        )
        
        return total_loss, {
            'total': total_loss.item(),
            'category': cat_loss.item(),
            'attribute': attr_loss.item(),
            'landmark': lm_loss.item()
        } 