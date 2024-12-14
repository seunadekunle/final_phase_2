import torch
import numpy as np
from data_loading import get_data_loader
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import os

def analyze_batch_distribution(loader, num_batches=10):
    """Analyze the distribution of categories and attributes in batches."""
    category_counts = defaultdict(int)
    attr_counts = defaultdict(int)
    attr_group_counts = defaultdict(lambda: defaultdict(int))
    total_samples = 0
    
    logging.info("Analyzing batch distributions...")
    
    # collect statistics from multiple batches
    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= num_batches:
            break
            
        # unpack batch data (handle both with and without landmarks)
        if len(batch_data) == 4:  # with landmarks
            images, attributes, categories, landmarks = batch_data
        else:  # without landmarks
            images, attributes, categories = batch_data
            
        # update category counts
        for cat in categories:
            category_counts[cat.item()] += 1
            
        # update attribute counts (sum over batch)
        attr_sums = attributes.sum(dim=0)
        for attr_idx, count in enumerate(attr_sums):
            attr_counts[attr_idx] += count.item()
            
        # update attribute group counts
        for group_id, group_indices in loader.dataset.attr_groups.items():
            group_mask = torch.zeros_like(attributes[0], dtype=torch.bool)
            group_mask[group_indices] = True
            group_attrs = attributes[:, group_mask]
            
            group_sums = group_attrs.sum(dim=0)
            for local_idx, count in enumerate(group_sums):
                attr_group_counts[group_id][local_idx] += count.item()
            
        total_samples += len(categories)
        
    return category_counts, attr_counts, attr_group_counts, total_samples

def plot_distributions(category_counts, attr_counts, attr_group_counts, total_samples, save_dir='src/experiments/sampling_analysis'):
    """Plot the distribution of categories and attributes."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set default style parameters for all plots
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.facecolor': '#f8f9fa'
    })
    
    # Plot category distribution
    plt.figure(figsize=(15, 5))
    categories = sorted(list(category_counts.keys()))
    counts = [category_counts[cat] / total_samples for cat in categories]
    
    plt.bar(categories, counts, alpha=0.8, color='#2196F3')
    plt.title('Category Distribution')
    plt.xlabel('Category ID')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir, 'category_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot overall attribute distribution
    plt.figure(figsize=(15, 5))
    attributes = sorted(list(attr_counts.keys()))
    counts = [attr_counts[attr] / total_samples for attr in attributes]
    
    plt.bar(attributes, counts, alpha=0.8, color='#4CAF50')
    plt.title('Overall Attribute Distribution')
    plt.xlabel('Attribute ID')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir, 'attribute_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot per-group attribute distributions
    for group_id, group_counts in attr_group_counts.items():
        plt.figure(figsize=(10, 4))
        attrs = sorted(list(group_counts.keys()))
        counts = [group_counts[attr] / total_samples for attr in attrs]
        
        plt.bar(range(len(attrs)), counts, alpha=0.8, color='#FF9800')
        plt.title(f'Group {group_id} Attribute Distribution')
        plt.xlabel('Attribute Index within Group')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(save_dir, f'group_{group_id}_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return os.path.join(save_dir, 'category_distribution.png'), os.path.join(save_dir, 'attribute_distribution.png')

def test_balanced_sampling():
    """Test the balanced sampling functionality."""
    # create data loaders with and without balanced sampling
    logging.info("Creating data loader with balanced sampling...")
    balanced_loader = get_data_loader(
        '../data', 
        'train',
        batch_size=32,
        subset_size=0.1,  # use subset for faster testing
        use_balanced_sampling=True,
        category_weight=0.5,
        attr_weight=0.5
    )
    
    logging.info("Creating data loader without balanced sampling...")
    regular_loader = get_data_loader(
        '../data',
        'train',
        batch_size=32,
        subset_size=0.1,
        use_balanced_sampling=False
    )
    
    # analyze distributions
    logging.info("Analyzing balanced loader distribution...")
    bal_cat_counts, bal_attr_counts, bal_group_counts, bal_total = analyze_batch_distribution(balanced_loader)
    
    logging.info("Analyzing regular loader distribution...")
    reg_cat_counts, reg_attr_counts, reg_group_counts, reg_total = analyze_batch_distribution(regular_loader)
    
    # plot distributions
    logging.info("Plotting distributions...")
    bal_plots = plot_distributions(
        bal_cat_counts, 
        bal_attr_counts,
        bal_group_counts,
        bal_total,
        save_dir='src/experiments/sampling_analysis/balanced'
    )
    reg_plots = plot_distributions(
        reg_cat_counts, 
        reg_attr_counts,
        reg_group_counts,
        reg_total,
        save_dir='src/experiments/sampling_analysis/regular'
    )
    
    # compute statistics
    bal_cat_std = np.std([count/bal_total for count in bal_cat_counts.values()])
    reg_cat_std = np.std([count/reg_total for count in reg_cat_counts.values()])
    
    bal_attr_std = np.std([count/bal_total for count in bal_attr_counts.values()])
    reg_attr_std = np.std([count/reg_total for count in reg_attr_counts.values()])
    
    # compute per-group statistics
    group_stats = []
    for group_id in bal_group_counts:
        bal_std = np.std([count/bal_total for count in bal_group_counts[group_id].values()])
        reg_std = np.std([count/reg_total for count in reg_group_counts[group_id].values()])
        improvement = ((reg_std - bal_std)/reg_std)*100 if reg_std > 0 else 0
        group_stats.append((group_id, bal_std, reg_std, improvement))
    
    # print results
    logging.info("\nSampling Analysis Results:")
    logging.info(f"{'='*50}")
    logging.info("Category Distribution Std Dev:")
    logging.info(f"  Balanced: {bal_cat_std:.4f}")
    logging.info(f"  Regular:  {reg_cat_std:.4f}")
    logging.info(f"  Improvement: {((reg_cat_std - bal_cat_std)/reg_cat_std)*100:.1f}%")
    
    logging.info("\nOverall Attribute Distribution Std Dev:")
    logging.info(f"  Balanced: {bal_attr_std:.4f}")
    logging.info(f"  Regular:  {reg_attr_std:.4f}")
    logging.info(f"  Improvement: {((reg_attr_std - bal_attr_std)/reg_attr_std)*100:.1f}%")
    
    logging.info("\nPer-Group Attribute Distribution Std Dev:")
    for group_id, bal_std, reg_std, improvement in group_stats:
        logging.info(f"Group {group_id}:")
        logging.info(f"  Balanced: {bal_std:.4f}")
        logging.info(f"  Regular:  {reg_std:.4f}")
        logging.info(f"  Improvement: {improvement:.1f}%")
    
    logging.info(f"\nPlots saved to:")
    logging.info(f"  Balanced: {bal_plots[0]}")
    logging.info(f"  Regular:  {reg_plots[0]}")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    test_balanced_sampling() 