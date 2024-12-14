"""Training Module For FashionNet Model.

This module provides training utilities and configurations for the FashionNet model,
including data loading, optimization, and evaluation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import json

from models.fashionnet import FashionNet
from models.losses import FashionNetLoss
from data_loading import get_data_loader
from evaluation import FashionNetEvaluator, log_metrics

def get_default_config():
    """Get Default Training Configuration For Full Dataset."""
    return {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'mps'),
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'loss_weights': {
            'category_weight': 1.0,
            'attribute_weight': 0.5,  # reduced weight for attributes
            'landmark_weight': 2.0,   # increased weight for landmarks
        },
        'scheduler': {
            'type': 'cosine',  # cosine annealing
            'T_max': 50,       # total epochs
            'eta_min': 1e-6    # minimum learning rate
        },
        'data_dir': 'data/deepfashion',
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'subset_size': 1.0  # use full dataset
    }

def get_debug_config():
    """Get Debug Configuration For Quick Testing."""
    # get absolute path to data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    # create log and checkpoint directories
    log_dir = os.path.join(os.path.dirname(current_dir), 'logs', 'debug_run')
    checkpoint_dir = os.path.join(os.path.dirname(current_dir), 'checkpoints', 'debug_run')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'mps'),
        'num_epochs': 10,           # increased epochs for better convergence
        'batch_size': 64,          # increased batch size for better statistics
        'learning_rate': 5e-4,     # increased learning rate
        'weight_decay': 1e-4,
        'loss_weights': {
            'category_weight': 10.0,    # significantly increased to focus on category accuracy
            'attribute_weight': 0.5,    # reduced since attribute performance is good
            'landmark_weight': 0.001    # significantly reduced due to high magnitude
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 10,            # match num_epochs
            'eta_min': 1e-5         # minimum learning rate
        },
        'root_dir': root_dir,      # absolute path to data directory
        'log_dir': log_dir,        # absolute path to log directory
        'checkpoint_dir': checkpoint_dir,  # absolute path to checkpoint directory
        'subset_size': 0.1,        # increased subset size for better training
        'max_grad_norm': 1.0,      # gradient clipping threshold
        'use_balanced_sampling': True,  # enable balanced sampling
        'optimizer': {
            'type': 'adamw',
            'betas': (0.9, 0.999),
            'group_lrs': {
                'backbone': 0.1,     # slower learning for backbone
                'category': 2.0,     # faster learning for category head
                'attribute': 1.0,    # normal learning for attribute heads
                'landmark': 0.5      # slower learning for landmark branch
            }
        }
    }

class FashionNetTrainer:
    """Trainer Class For FashionNet Model With Multi-Task Learning."""
    
    def __init__(self, config):
        """Initialize FashionNet Trainer With Given Configuration."""
        self.config = config
        self.device = config['device']
        
        # create directories for logging and checkpoints
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # initialize tensorboard writer
        self.writer = SummaryWriter(config['log_dir'])
        
        # save configuration
        config_save = config.copy()
        config_save['device'] = str(config_save['device'])
        with open(os.path.join(config['log_dir'], 'config.json'), 'w') as f:
            json.dump(config_save, f, indent=4)
        
        # initialize model components
        self._initialize_model()
        self._initialize_loss()
        self._initialize_optimizer()
        self._initialize_scheduler()
        self._initialize_dataloaders()
        
        # initialize training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_metrics = None
    
    def _initialize_model(self):
        """Initialize And Configure FashionNet Model."""
        self.model = FashionNet(
            num_categories=50,
            attr_group_dims=[7,3,3,4,6,3],
            num_landmarks=8,
            pretrained=True
        ).to(self.device)
    
    def _initialize_loss(self):
        """Initialize Loss Functions With Configured Weights."""
        self.criterion = FashionNetLoss(
            category_weight=self.config['loss_weights']['category_weight'],
            attribute_weight=self.config['loss_weights']['attribute_weight'],
            landmark_weight=self.config['loss_weights']['landmark_weight']
        ).to(self.device)
    
    def _initialize_optimizer(self):
        """Initialize Optimizer With Layer-Specific Learning Rates."""
        self.optimizer = optim.AdamW([
            {'params': self.model.features.parameters(), 'lr': self.config['learning_rate'] * 0.1},  # slower for backbone
            {'params': self.model.landmark_branch.parameters(), 'lr': self.config['learning_rate']},
            {'params': self.model.category_branch.parameters(), 'lr': self.config['learning_rate'] * 2.0},  # faster for category head
            {'params': self.model.category_attention.parameters(), 'lr': self.config['learning_rate'] * 2.0},  # attention also faster
            {'params': self.model.attr_groups.parameters(), 'lr': self.config['learning_rate']},
            {'params': self.model.fusion.parameters(), 'lr': self.config['learning_rate']}
        ], weight_decay=self.config['weight_decay'])
    
    def _initialize_scheduler(self):
        """Initialize Learning Rate Scheduler."""
        if self.config['scheduler']['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['scheduler']['T_max'],
                eta_min=self.config['scheduler']['eta_min']
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config['scheduler']['type']}")
    
    def _initialize_dataloaders(self):
        """Initialize Train And Validation Data Loaders."""
        self.train_loader = get_data_loader(
            root_dir=self.config['root_dir'],
            split='train',
            batch_size=self.config['batch_size'],
            num_workers=2,  # reduced workers for CPU
            subset_size=self.config['subset_size'],
            use_balanced_sampling=self.config.get('use_balanced_sampling', False)
        )
        
        self.val_loader = get_data_loader(
            root_dir=self.config['root_dir'],
            split='val',
            batch_size=self.config['batch_size'],
            num_workers=2,  # reduced workers for CPU
            subset_size=self.config['subset_size']
        )
        
        # initialize evaluator
        self.evaluator = FashionNetEvaluator(self.model, self.val_loader, self.device)
    
    def train_epoch(self):
        """Train Model For One Epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {'category': 0, 'attribute': 0, 'landmark': 0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (images, attributes, categories, landmarks) in enumerate(pbar):
            # prepare data
            images = images.to(self.device)
            attributes = attributes.to(self.device)
            categories = categories.to(self.device)
            landmark_coords = landmarks.to(self.device)
            
            # create visibility flags (assume all landmarks are visible for now)
            landmark_vis = torch.ones(landmarks.shape[0], landmarks.shape[1]).to(self.device)
            
            # forward pass and loss computation
            self.optimizer.zero_grad()
            category_pred, attr_preds, landmark_pred = self.model(images)
            loss, losses = self.criterion(
                (category_pred, attr_preds, landmark_pred),
                (categories, attributes, landmark_coords, landmark_vis)
            )
            
            # backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            
            # update metrics
            total_loss += loss.item()
            for k, v in losses.items():
                if k != 'total':
                    loss_components[k] += v
            
            # update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cat': f'{losses["category"]:.4f}',
                'attr': f'{losses["attribute"]:.4f}',
                'lm': f'{losses["landmark"]:.4f}'
            })
            
            # log batch metrics
            step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', loss.item(), step)
            for k, v in losses.items():
                if k != 'total':
                    self.writer.add_scalar(f'train/batch_{k}_loss', v, step)
            
            # log learning rates at start of epoch
            if batch_idx == 0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f'train/lr_group_{i}', param_group['lr'], self.current_epoch)
            
            num_batches += 1
        
        # compute epoch metrics
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_components = {k: v / num_batches for k, v in loss_components.items()}
        else:
            avg_loss = float('inf')
            avg_components = {k: float('inf') for k in loss_components}
            logging.warning('No batches processed in training epoch')
        
        return avg_loss, avg_components
    
    def validate(self):
        """Run Validation."""
        self.model.eval()
        total_loss = 0
        loss_components = {'category': 0, 'attribute': 0, 'landmark': 0}
        
        with torch.no_grad():
            for images, attributes, categories, landmarks in tqdm(self.val_loader, desc='Validation'):
                # move data to device
                images = images.to(self.device)
                attributes = attributes.to(self.device)
                categories = categories.to(self.device)
                landmark_coords = landmarks.to(self.device)  # [batch_size, num_landmarks, 2]
                
                # create dummy visibility flags (assume all landmarks are visible)
                landmark_vis = torch.ones(landmarks.shape[0], landmarks.shape[1]).to(self.device)
                
                # forward pass
                category_pred, attr_preds, landmark_pred = self.model(images)
                
                # compute loss
                loss, losses = self.criterion(
                    (category_pred, attr_preds, landmark_pred),
                    (categories, attributes, landmark_coords, landmark_vis)
                )
                
                # update metrics
                total_loss += loss.item()
                for k, v in losses.items():
                    if k != 'total':
                        loss_components[k] += v
        
        # compute epoch metrics
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        # compute evaluation metrics
        metrics = self.evaluator.evaluate()
        
        # log validation metrics
        self.writer.add_scalar('val/epoch_loss', avg_loss, self.current_epoch)
        for k, v in avg_components.items():
            self.writer.add_scalar(f'val/{k}_loss', v, self.current_epoch)
        
        # log evaluation metrics
        for k, v in metrics.items():
            self.writer.add_scalar(f'val/{k}', v, self.current_epoch)
        
        return avg_loss, avg_components, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save Checkpoint Of The Model.
        
        Args:
            epoch (int): Current Epoch Number
            is_best (bool): Whether This Is The Best Model So Far
        """
        try:
            # create checkpoint directory if it doesn't exist
            os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
            
            # prepare checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config
            }
            
            # save regular checkpoint
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            
            # save best model if needed
            if is_best:
                best_model_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
                # save best model state dict only to reduce file size
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"New best model saved at epoch {epoch}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
            # try saving only model state dict as fallback
            try:
                fallback_path = os.path.join(self.config['checkpoint_dir'], f'model_state_epoch_{epoch}.pth')
                torch.save(self.model.state_dict(), fallback_path)
                logging.info(f"Saved model state dict as fallback at {fallback_path}")
            except Exception as e:
                logging.error(f"Error saving model state dict: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load Checkpoint.
        
        Args:
            checkpoint_path (str): Path To Checkpoint File
        """
        try:
            # check if file exists
            if not os.path.exists(checkpoint_path):
                logging.warning(f"Checkpoint {checkpoint_path} does not exist")
                return
            

            checkpoint = torch.load(checkpoint_path, map_location='mps')
            
            # load model state
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # full checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.current_epoch = checkpoint['epoch']
                self.best_val_loss = checkpoint['best_val_loss']
            else:
                # state dict only
                self.model.load_state_dict(checkpoint)
            
            logging.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
    
    def train(self):
        """Main Training Loop."""
        logging.info("Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            logging.info(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # train for one epoch
            train_loss, train_components = self.train_epoch()
            logging.info(f"Training loss: {train_loss:.4f}")
            for k, v in train_components.items():
                logging.info(f"Training {k} loss: {v:.4f}")
            
            # validate
            val_loss, val_components, metrics = self.validate()
            logging.info(f"Validation loss: {val_loss:.4f}")
            for k, v in val_components.items():
                logging.info(f"Validation {k} loss: {v:.4f}")
            
            # log evaluation metrics
            log_metrics(metrics, split='val')
            
            # update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingLR):
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)
            
            # save checkpoint if best so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # regular checkpoint save
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)

def main():
    """Main Training Function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config = get_debug_config()  
    
    trainer = FashionNetTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main() 