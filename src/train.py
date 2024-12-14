import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.attribute_classifier import AttributeClassifier
from models.losses import AttributeLoss
from data_loading import get_data_loader

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    # setup progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, attributes) in enumerate(pbar):
        images, attributes = images.to(device), attributes.to(device)
        
        # forward pass
        logits = model(images)
        
        # compute loss
        loss = criterion(logits, attributes)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update metrics
        total_loss += loss.item()
        
        # update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
    
    # compute epoch average
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    attr_preds = []
    attr_targets = []
    
    with torch.no_grad():
        for images, attributes in val_loader:
            images, attributes = images.to(device), attributes.to(device)
            
            # forward pass
            logits = model(images)
            
            # compute loss
            loss = criterion(logits, attributes)
            
            # update metrics
            total_loss += loss.item()
            
            # store predictions and targets for metrics
            attr_preds.append(torch.sigmoid(logits).cpu())
            attr_targets.append(attributes.cpu())
    
    # compute average loss
    avg_loss = total_loss / len(val_loader)
    
    # compute attribute prediction metrics
    attr_preds = torch.cat(attr_preds, dim=0)
    attr_targets = torch.cat(attr_targets, dim=0)
    
    # compute top-k accuracy for k in [3, 5]
    metrics = {}
    for k in [3, 5]:
        top_k_acc = compute_top_k_recall(attr_preds, attr_targets, k)
        metrics[f'top_{k}_recall'] = top_k_acc
    
    return avg_loss, metrics

def compute_top_k_recall(predictions, targets, k):
    """Compute top-k recall for multi-label prediction."""
    batch_size = predictions.size(0)
    
    # get top k predictions for each sample
    _, top_k_indices = torch.topk(predictions, k, dim=1)
    
    # create a mask of correct predictions
    correct = torch.zeros_like(predictions, dtype=torch.bool)
    for i in range(batch_size):
        correct[i, top_k_indices[i]] = True
    
    # compute recall
    recall = (correct & (targets == 1)).sum().float() / targets.sum()
    
    return recall.item()

def main(args):
    # get absolute path for data directory
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(args.data_dir)
    
    logger.info(f'Using data directory: {args.data_dir}')
    
    # verify data files exist
    train_file = os.path.join(args.data_dir, 'Anno_fine', 'train.txt')
    train_attr_file = os.path.join(args.data_dir, 'Anno_fine', 'train_attr.txt')
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f'Training split file not found: {train_file}')
    if not os.path.exists(train_attr_file):
        raise FileNotFoundError(f'Training attributes file not found: {train_attr_file}')
        
    logger.info('Found all required data files')
    
    # create save directory
    save_dir = os.path.join('experiments', 'baseline_runs', 
                           datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f'Created save directory: {save_dir}')
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    logger.info(f'Using device: {device}')
    
    # create data loaders
    logger.info('Creating data loaders...')
    train_loader = get_data_loader(args.data_dir, 'train', args.batch_size, subset_size=args.subset_size)
    val_loader = get_data_loader(args.data_dir, 'val', args.batch_size, subset_size=args.subset_size)
    logger.info('Data loaders created successfully')
    
    # create model
    model = AttributeClassifier(num_attributes=26)
    model = model.to(device)
    logger.info('Model created and moved to device')
    
    # create loss function
    criterion = AttributeLoss()
    
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                          weight_decay=args.weight_decay)
    
    # training loop
    best_recall = 0
    logger.info('Starting training...')
    for epoch in range(1, args.epochs + 1):
        # train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # log metrics
        logger.info(
            f'Epoch {epoch} - '
            f'Train loss: {train_loss:.4f} | '
            f'Val loss: {val_loss:.4f}, '
            f'top_3_recall: {val_metrics["top_3_recall"]:.4f}, '
            f'top_5_recall: {val_metrics["top_5_recall"]:.4f}'
        )
        
        # save best model
        if val_metrics['top_3_recall'] > best_recall:
            best_recall = val_metrics['top_3_recall']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall': best_recall,
                'val_metrics': val_metrics
            }, os.path.join(save_dir, 'best_model.pth'))
            logger.info(f'Saved new best model with recall: {best_recall:.4f}')
            
        # save checkpoint every N epochs
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            logger.info(f'Saved checkpoint for epoch {epoch}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Attribute Classifier')
    
    # data args
    parser.add_argument('--data_dir', type=str, default='data',
                        help='path to dataset root directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size')
    parser.add_argument('--subset_size', type=float, default=0.1,
                        help='fraction of dataset to use (0.0 to 1.0)')
    
    # training args
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='frequency of saving checkpoints')
    
    args = parser.parse_args()
    main(args) 