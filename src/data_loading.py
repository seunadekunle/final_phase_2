import os
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import logging
import numpy as np
from collections import defaultdict
from typing import Optional, List
import random

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BalancedBatchSampler(Sampler):
    """Sampler that returns balanced batches of data."""
    
    def __init__(self, dataset, batch_size=32):
        """Initialize the sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Number of samples per batch
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        
        # group indices by category
        self.category_indices = defaultdict(list)
        for idx, category in enumerate(dataset.categories):
            self.category_indices[category].append(idx)
        
        # compute samples per category to fill batch
        num_categories = len(self.category_indices)
        self.samples_per_category = max(1, batch_size // num_categories)
        
        # compute number of complete batches possible
        min_samples_per_category = min(len(indices) for indices in self.category_indices.values())
        batches_per_category = min_samples_per_category // self.samples_per_category
        self.num_batches = max(1, batches_per_category)  # ensure at least one batch
        
        # compute total number of samples
        self.total_samples = self.num_batches * num_categories * self.samples_per_category
        
        logging.info(f"Created balanced sampler with {self.num_batches} batches")
        logging.info(f"Samples per category per batch: {self.samples_per_category}")
        logging.info(f"Number of categories: {num_categories}")
        logging.info(f"Total samples: {self.total_samples}")
        logging.info(f"Min samples per category: {min_samples_per_category}")
    
    def __iter__(self):
        # shuffle indices within each category
        for category in self.category_indices:
            random.shuffle(self.category_indices[category])
        
        # create balanced batches
        current_indices = {k: 0 for k in self.category_indices}
        batches = []
        
        for _ in range(self.num_batches):
            batch = []
            # sample equally from each category
            categories = list(self.category_indices.keys())
            random.shuffle(categories)
            
            for category in categories:
                # get next indices for this category
                start_idx = current_indices[category]
                end_idx = start_idx + self.samples_per_category
                
                # handle case where we don't have enough samples
                if end_idx > len(self.category_indices[category]):
                    # reshuffle and start from beginning
                    random.shuffle(self.category_indices[category])
                    current_indices[category] = 0
                    start_idx = 0
                    end_idx = self.samples_per_category
                
                indices = self.category_indices[category][start_idx:end_idx]
                batch.extend(indices)
                current_indices[category] = end_idx
            
            # shuffle batch
            random.shuffle(batch)
            batches.append(batch)
        
        return iter(batches)
    
    def __len__(self):
        return self.num_batches

class DeepFashionDataset(Dataset):
    """DeepFashion dataset."""
    
    def __init__(self, root_dir, split='train', subset_size=1.0):
        """Initialize the dataset.
        
        Args:
            root_dir (str): Path to dataset directory
            split (str): One of ['train', 'val', 'test']
            subset_size (float): Fraction of dataset to use (for debugging)
        """
        self.root_dir = root_dir
        self.split = split
        self.subset_size = subset_size
        
        # setup augmentation
        if split == 'train':
            self.transform = transforms.Compose([
                # resize with padding to maintain aspect ratio
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
                
                # convert to tensor first
                transforms.ToTensor(),
                
                # color augmentation
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                
                # random augmentations
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                
                # advanced augmentations
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3)
                ], p=0.3),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
                
                # cutout augmentation
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
                
                # normalize
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # load data
        self.attr_types = self._load_attribute_types()
        self.attr_type_map = self._create_attr_type_mapping()
        self.categories = []
        self.category_names = []
        self._load_categories()
        self.image_paths = []
        self.attributes = []
        self.landmarks = []
        self._load_annotations()
        
        # apply subset sampling if needed
        if subset_size < 1.0:
            num_samples = len(self.image_paths)
            subset_size = int(num_samples * subset_size)
            indices = torch.randperm(num_samples)[:subset_size]
            
            self.image_paths = [self.image_paths[i] for i in indices]
            self.attributes = self.attributes[indices]
            self.landmarks = self.landmarks[indices]
            self.categories = [self.categories[i] for i in indices]
    
    def _load_attribute_types(self):
        """Load attribute types from file."""
        attr_types = []
        attr_type_file = os.path.join(self.root_dir, 'Anno_fine', 'list_attr_cloth.txt')
        
        with open(attr_type_file, 'r') as f:
            lines = f.readlines()
            num_attrs = int(lines[0].strip())
            for line in lines[2:2+num_attrs]:  # skip header lines
                attr_type = line.strip().split()[0]  # take only the attribute name
                attr_types.append(attr_type)
        
        return attr_types
    
    def _create_attr_type_mapping(self):
        """Create mapping from attribute type to index range."""
        attr_type_map = {}
        current_idx = 0
        
        for attr_type in self.attr_types:
            num_attrs = len(attr_type.split())
            attr_type_map[attr_type] = (current_idx, current_idx + num_attrs)
            current_idx += num_attrs
        
        return attr_type_map
    
    def _load_categories(self):
        """Load category information."""
        category_file = os.path.join(self.root_dir, 'Anno_fine', 'list_category_cloth.txt')
        
        with open(category_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # skip header lines
                category = line.strip()
                self.category_names.append(category)
    
    def _load_annotations(self):
        """Load image paths and annotations."""
        # load split-specific files
        split_file = os.path.join(self.root_dir, 'Anno_fine', f'{self.split}.txt')
        attr_file = os.path.join(self.root_dir, 'Anno_fine', f'{self.split}_attr.txt')
        landmark_file = os.path.join(self.root_dir, 'Anno_fine', f'{self.split}_landmarks.txt')
        category_file = os.path.join(self.root_dir, 'Anno_fine', f'{self.split}_cate.txt')
        
        # load image paths
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # skip header lines
                img_path = line.strip()
                # Remove 'img/' prefix if it exists
                if img_path.startswith('img/'):
                    img_path = img_path[4:]
                self.image_paths.append(os.path.join(self.root_dir, 'img', img_path))
        
        # load attributes
        with open(attr_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # skip header lines
                attrs = line.strip().split()
                attrs = [1 if a == '1' else 0 for a in attrs]
                self.attributes.append(attrs)
        self.attributes = torch.tensor(self.attributes, dtype=torch.float32)
        
        # load landmarks
        with open(landmark_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # skip header lines
                coords = line.strip().split()
                coords = [float(c) for c in coords]
                # reshape to (8, 2) format
                coords = torch.tensor(coords, dtype=torch.float32)
                coords = coords.view(-1, 2)  # reshape to (8, 2)
                self.landmarks.append(coords)
        self.landmarks = torch.stack(self.landmarks)
        
        # load categories
        with open(category_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:  # skip header lines
                category = int(line.strip())
                self.categories.append(category)
        
        # verify data loading
        num_samples = len(self.image_paths)
        assert len(self.attributes) == num_samples, f"Mismatch in number of samples: {len(self.attributes)} attributes vs {num_samples} images"
        assert len(self.landmarks) == num_samples, f"Mismatch in number of samples: {len(self.landmarks)} landmarks vs {num_samples} images"
        assert len(self.categories) == num_samples, f"Mismatch in number of samples: {len(self.categories)} categories vs {num_samples} images"
        
        logging.info(f"Loaded {num_samples} samples for {self.split} split")
        logging.info(f"First image path: {self.image_paths[0]}")
        logging.info(f"Number of attributes: {self.attributes.shape[1]}")
        logging.info(f"Number of landmarks: {self.landmarks.shape[1]}")
        logging.info(f"Number of categories: {len(set(self.categories))}")
    
    def __len__(self):
        """Return the size of dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get item by index."""
        # load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # apply transforms
        image = self.transform(image)
        
        return (
            image,
            self.attributes[idx],
            self.categories[idx],
            self.landmarks[idx]
        )

def balanced_collate_fn(batch):
    """Collate function for balanced batch sampler."""
    return batch[0]

def get_data_loader(root_dir, split='train', batch_size=32, num_workers=4, 
                  subset_size=1.0, use_balanced_sampling=False):
    """Get data loader for DeepFashion dataset.
    
    Args:
        root_dir (str): Path to dataset directory
        split (str): One of ['train', 'val', 'test']
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        subset_size (float): Fraction of dataset to use (for debugging)
        use_balanced_sampling (bool): Whether to use balanced sampling
        
    Returns:
        torch.utils.data.DataLoader: Data loader
    """
    # Create dataset
    dataset = DeepFashionDataset(
        root_dir=root_dir,
        split=split,
        subset_size=subset_size
    )
    
    # Create sampler if using balanced sampling
    sampler = None
    if use_balanced_sampling and split == 'train':
        # Create indices for each category
        category_indices = defaultdict(list)
        for idx, category in enumerate(dataset.categories):
            category_indices[category].append(idx)
        
        # Compute samples per category
        num_categories = len(category_indices)
        samples_per_category = max(1, batch_size // num_categories)
        
        # Create balanced indices
        indices = []
        for category in category_indices:
            # Get indices for this category
            cat_indices = category_indices[category]
            # Repeat indices if needed
            while len(cat_indices) < samples_per_category * 100:  # ensure enough samples
                cat_indices = cat_indices * 2
            # Sample randomly
            sampled_indices = random.sample(cat_indices, samples_per_category * 100)
            indices.extend(sampled_indices)
        
        # Shuffle indices
        random.shuffle(indices)
        
        # Create sampler
        sampler = SubsetRandomSampler(indices)
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True
    )
    
    return loader 