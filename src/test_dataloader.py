import os
from data_loading import get_data_loader

def test_dataloader():
    # get current directory and go up one level to find data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    # test train loader
    train_loader = get_data_loader(root_dir, 'train', batch_size=4)
    
    # get first batch
    images, attributes = next(iter(train_loader))
    
    print(f'Batch shape: {images.shape}')
    print(f'Attributes shape: {attributes.shape}')
    print(f'Sample attribute vector: {attributes[0]}')

if __name__ == '__main__':
    test_dataloader() 