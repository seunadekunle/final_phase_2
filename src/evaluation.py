import torch
import numpy as np
from tqdm import tqdm
import logging
import torch.nn.functional as F

def compute_topk_accuracy(pred, target, k=5):
    """Compute top-k accuracy.
    
    Args:
        pred (torch.Tensor): Predicted logits [batch_size, num_classes]
        target (torch.Tensor): Target labels [batch_size]
        k (int): Number of top predictions to consider
        
    Returns:
        float: Top-k accuracy
    """
    batch_size = target.size(0)
    _, pred = pred.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size).item()

def compute_topk_recall(pred, target, k=5):
    """Compute top-k recall for multi-label prediction.
    
    Args:
        pred (torch.Tensor): Predicted logits [batch_size, num_attributes]
        target (torch.Tensor): Target binary labels [batch_size, num_attributes]
        k (int): Number of top predictions to consider
        
    Returns:
        float: Top-k recall
    """
    batch_size = target.size(0)
    num_attributes = target.size(1)
    
    # adjust k if it's larger than number of attributes
    k = min(k, num_attributes)
    
    # apply sigmoid to get probabilities
    pred_probs = torch.sigmoid(pred)
    
    # get top k predictions per sample
    _, top_indices = pred_probs.topk(k, dim=1)
    
    total_recall = 0
    total_samples = 0
    
    for i in range(batch_size):
        # get positive attributes for this sample
        true_positives = torch.where(target[i] == 1)[0]
        if len(true_positives) > 0:
            # get predictions for this sample
            sample_preds = set(top_indices[i].cpu().numpy())
            sample_true = set(true_positives.cpu().numpy())
            
            # compute recall for this sample
            recall = len(sample_preds.intersection(sample_true)) / len(sample_true)
            total_recall += recall
            total_samples += 1
    
    # return average recall across samples with positive attributes
    return (total_recall / total_samples * 100) if total_samples > 0 else 0.0

def compute_attribute_metrics(predictions, targets, k_values=[3, 5]):
    """Compute attribute prediction metrics.
    
    Args:
        predictions (list): List of predicted logits for each attribute group
        targets (torch.Tensor): Target binary labels [batch_size, total_attributes]
        k_values (list): List of k values for top-k metrics
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    start_idx = 0
    
    # compute metrics for each group
    group_recalls = []
    for group_pred in predictions:
        num_attrs = group_pred.size(1)
        group_target = targets[:, start_idx:start_idx + num_attrs]
        
        # compute metrics only for positive samples
        pos_samples = (group_target == 1).any(dim=1)
        if pos_samples.any():
            group_pred = group_pred[pos_samples]
            group_target = group_target[pos_samples]
            
            # compute recall@k for each k
            for k in k_values:
                # get top k predictions
                _, top_k = torch.topk(group_pred, min(k, group_pred.size(1)), dim=1)
                
                # create mask of correct predictions
                correct = torch.zeros_like(group_pred, dtype=torch.bool)
                for i, top_indices in enumerate(top_k):
                    correct[i, top_indices] = True
                
                # compute recall only for positive samples
                recall = (correct & (group_target == 1)).sum().float() / (group_target == 1).sum().float()
                group_recalls.append(recall.item())
        
        start_idx += num_attrs
    
    # average recalls across groups
    for k_idx, k in enumerate(k_values):
        k_recalls = group_recalls[k_idx::len(k_values)]  # get recalls for this k
        metrics[f'recall@{k}'] = np.mean(k_recalls) * 100
    
    return metrics

def compute_landmark_metrics(predictions, targets, visibility=None, threshold=0.1):
    """Compute landmark prediction metrics.
    
    Args:
        predictions (torch.Tensor): Predicted landmarks [batch_size, num_landmarks, 3]
        targets (torch.Tensor): Target landmarks [batch_size, num_landmarks, 2]
        visibility (torch.Tensor): Target visibility [batch_size, num_landmarks]
        threshold (float): Distance threshold for detection rate
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # extract coordinates and predicted visibility
    pred_coords = predictions[:, :, :2]
    pred_vis = torch.sigmoid(predictions[:, :, 2])
    
    # compute MSE for visible landmarks
    if visibility is None:
        visibility = torch.ones_like(pred_vis)
    
    mse = F.mse_loss(
        pred_coords * visibility.unsqueeze(-1),
        targets * visibility.unsqueeze(-1)
    )
    metrics['mse'] = mse.item()
    
    # compute detection rate
    distances = torch.norm(pred_coords - targets, dim=2)  # [batch_size, num_landmarks]
    detected = (distances < threshold) & (pred_vis > 0.5) & (visibility == 1)
    detection_rate = detected.float().mean().item() * 100
    metrics['detection_rate'] = detection_rate
    
    return metrics

class FashionNetEvaluator:
    """Evaluator for FashionNet model."""
    
    def __init__(self, model, val_loader, device):
        """Initialize evaluator.
        
        Args:
            model (nn.Module): Model to evaluate
            val_loader (DataLoader): Validation data loader
            device (torch.device): Device to use
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
    
    def evaluate(self):
        """Evaluate model on validation set.
        
        Returns:
            dict: Dictionary of metrics
        """
        self.model.eval()
        metrics = {
            'category_top3': 0,
            'category_top5': 0,
            'recall@3': 0,
            'recall@5': 0,
            'mse': 0,
            'detection_rate': 0
        }
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Evaluating')
            for images, attributes, categories, landmarks in pbar:
                # move data to device
                images = images.to(self.device)
                attributes = attributes.to(self.device)
                categories = categories.to(self.device)
                landmarks = landmarks.to(self.device)
                
                # forward pass
                category_pred, attr_preds, landmark_pred = self.model(images)
                
                # compute metrics
                metrics['category_top3'] += compute_topk_accuracy(category_pred, categories, k=3)
                metrics['category_top5'] += compute_topk_accuracy(category_pred, categories, k=5)
                
                # compute attribute metrics
                attr_metrics = compute_attribute_metrics(attr_preds, attributes)
                for k, v in attr_metrics.items():
                    metrics[k] += v
                
                # compute landmark metrics
                lm_metrics = compute_landmark_metrics(landmark_pred, landmarks)
                metrics['mse'] += lm_metrics['mse']
                metrics['detection_rate'] += lm_metrics['detection_rate']
                
                num_batches += 1
        
        # average metrics
        for k in metrics:
            metrics[k] /= num_batches
        
        return metrics

def log_metrics(metrics, split='val'):
    """Log evaluation metrics."""
    logging.info(f"\n{split.capitalize()} Metrics:")
    logging.info(f"Category Classification:")
    logging.info(f"  Top-3 Accuracy: {metrics['category_top3']:.2f}%")
    logging.info(f"  Top-5 Accuracy: {metrics['category_top5']:.2f}%")
    logging.info(f"Attribute Prediction:")
    logging.info(f"  Top-3 Recall: {metrics['recall@3']:.2f}%")
    logging.info(f"  Top-5 Recall: {metrics['recall@5']:.2f}%")
    logging.info(f"Landmark Prediction:")
    logging.info(f"  MSE: {metrics['mse']:.4f}")
    logging.info(f"  Detection Rate: {metrics['detection_rate']:.2f}%") 