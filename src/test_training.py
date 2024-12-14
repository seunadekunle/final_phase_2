import logging
import os
from train_fashionnet import FashionNetTrainer

def test_training():
    """Test the FashionNet training pipeline."""
    logging.info("Testing training pipeline...")
    
    # get absolute path to data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    # create minimal config for testing
    config = {
        'data_dir': data_dir,  # use absolute path
        'batch_size': 4,
        'num_epochs': 1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'subset_size': 0.01,  # use only 1% of data for testing
        'loss_weights': {
            'category_weight': 1.0,
            'attribute_weight': 1.0,
            'landmark_weight': 0.5
        },
        'device': 'mps',
        'log_dir': os.path.join(current_dir, 'experiments/test_run'),
        'checkpoint_dir': os.path.join(current_dir, 'experiments/test_run/checkpoints')
    }
    
    # create trainer
    trainer = FashionNetTrainer(config)
    
    # run one epoch
    logging.info("Running one training epoch...")
    train_loss, train_components = trainer.train_epoch()
    
    logging.info(f"\nTraining metrics:")
    logging.info(f"Total loss: {train_loss:.4f}")
    for k, v in train_components.items():
        logging.info(f"{k.capitalize()} loss: {v:.4f}")
    
    # run validation
    logging.info("\nRunning validation...")
    val_loss, val_components = trainer.validate()
    
    logging.info(f"\nValidation metrics:")
    logging.info(f"Total loss: {val_loss:.4f}")
    for k, v in val_components.items():
        logging.info(f"{k.capitalize()} loss: {v:.4f}")
    
    logging.info("\nTest completed successfully!")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    test_training() 