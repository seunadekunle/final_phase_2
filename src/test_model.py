import torch
from models.darn import DARN
from data_loading import get_data_loader

def test_model():
    # create model instance
    model = DARN(num_attributes=26, embedding_dim=512)
    model = model.train()  # set to training mode
    
    # get a batch of data
    data_loader = get_data_loader('../data', batch_size=4)
    images, attributes = next(iter(data_loader))
    
    # forward pass
    attr_logits, embedding = model(images)
    
    # print shapes and stats
    print(f'Input shape: {images.shape}')
    print(f'Attribute logits shape: {attr_logits.shape}')
    print(f'Embedding shape: {embedding.shape}')
    print(f'Embedding norm: {torch.norm(embedding[0])}')  # should be close to 1
    
    # test attribute prediction
    attr_probs = torch.sigmoid(attr_logits)
    print(f'\nSample attribute probabilities:\n{attr_probs[0]}\n')
    print(f'Ground truth attributes:\n{attributes[0]}')

if __name__ == '__main__':
    test_model() 