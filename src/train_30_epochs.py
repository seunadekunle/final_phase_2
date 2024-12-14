"""Training Script For 30-Epoch FashionNet Training.

This script configures and runs a 30-epoch training session for the FashionNet model,
with optimized hyperparameters and logging.
"""

import os
import torch
from train_fashionnet import FashionNetTrainer

def get_30_epoch_config():
    """Get Optimized Configuration For 30-Epoch Training Run.
    
    Returns:
        dict: Configuration dictionary with optimized parameters
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    # create log and checkpoint directories
    log_dir = os.path.join(os.path.dirname(current_dir), 'logs', '30_epoch_run')
    checkpoint_dir = os.path.join(os.path.dirname(current_dir), 'checkpoints', '30_epoch_run')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return {
        'device': torch.device('mps' if torch.cuda.is_available() else 'cpu'),
        'num_epochs': 30,           # 30 epochs training
        'batch_size': 32,          # reduced batch size for CPU training
        'learning_rate': 1e-4,     # reduced learning rate for stability
        'weight_decay': 1e-4,      # L2 regularization
        'loss_weights': {
            'category_weight': 1.0,     # base weight for category
            'attribute_weight': 0.5,    # reduced weight for attributes
            'landmark_weight': 0.1      # significantly reduced for stability
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 30,            # match num_epochs
            'eta_min': 1e-6         # minimum learning rate
        },
        'root_dir': root_dir,
        'log_dir': log_dir,
        'checkpoint_dir': checkpoint_dir,
        'subset_size': 0.1,        # using 10% of dataset for faster training
        'max_grad_norm': 1.0,      # gradient clipping threshold
        'use_balanced_sampling': True,  # enable balanced sampling
        'optimizer': {
            'type': 'adamw',
            'betas': (0.9, 0.999),
            'group_lrs': {
                'backbone': 0.1,     # slower learning for backbone
                'category': 1.0,     # normal learning for category head
                'attribute': 0.5,    # slower for attributes
                'landmark': 0.1      # much slower for landmarks
            }
        }
    }

if __name__ == '__main__':
    # set up logging configuration
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # initialize training components
        config = get_30_epoch_config()
        trainer = FashionNetTrainer(config)
        
        # training loop
        best_val_loss = float('inf')
        for epoch in range(config['num_epochs']):
            logging.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            
            # train for one epoch
            train_loss, train_components = trainer.train_epoch()
            logging.info(f"Training Loss: {train_loss:.4f}")
            logging.info(f"Components - Category: {train_components['category']:.4f}, "
                        f"Attribute: {train_components['attribute']:.4f}, "
                        f"Landmark: {train_components['landmark']:.4f}")
            
            # validate and compute metrics
            val_metrics = trainer.evaluator.evaluate()
            logging.info(f"Validation Metrics:")
            logging.info(f"Category Top-3: {val_metrics.get('category_top3', 0):.2f}%")
            logging.info(f"Category Top-5: {val_metrics.get('category_top5', 0):.2f}%")
            logging.info(f"Attribute Recall@3: {val_metrics.get('recall@3', 0):.2f}%")
            logging.info(f"Attribute Recall@5: {val_metrics.get('recall@5', 0):.2f}%")
            
            # save checkpoint if validation improved
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    config['checkpoint_dir'], 
                    f'best_model_epoch_{epoch + 1}.pth'
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': config
                }, checkpoint_path)
                logging.info(f"Saved best model checkpoint to {checkpoint_path}")
            
            # update learning rate
            trainer.scheduler.step()
            
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user. Saving checkpoint...")
        checkpoint_path = os.path.join(config['checkpoint_dir'], 'interrupted_checkpoint.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'val_metrics': val_metrics if 'val_metrics' in locals() else None,
            'config': config
        }, checkpoint_path)
        logging.info(f"Saved interrupt checkpoint to {checkpoint_path}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise