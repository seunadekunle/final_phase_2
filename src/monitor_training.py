import os
import time
import json
from datetime import datetime
import logging
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def monitor_training(log_dir='logs/debug_run'):
    """Monitor training progress from tensorboard logs.
    
    Args:
        log_dir (str): Directory containing tensorboard logs
    """
    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # initialize event accumulator
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # get available tags
    tags = event_acc.Tags()['scalars']
    
    # print latest metrics
    logging.info("\nLatest Training Metrics:")
    logging.info("-" * 50)
    
    for tag in tags:
        events = event_acc.Scalars(tag)
        if events:
            latest = events[-1]
            logging.info(f"{tag:30s}: {latest.value:.4f}")
    
    # plot loss curves
    plt.figure(figsize=(12, 6))
    
    # training losses
    for tag in [t for t in tags if 'train' in t and 'loss' in t]:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.plot(steps, values, label=tag)
    
    plt.title('Training Losses')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # save plot
    plot_dir = os.path.join(os.path.dirname(log_dir), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'training_losses.png'))
    logging.info(f"\nLoss plot saved to {plot_dir}/training_losses.png")

def main():
    """Main monitoring function."""
    while True:
        monitor_training()
        time.sleep(60)  # update every minute

if __name__ == '__main__':
    main() 