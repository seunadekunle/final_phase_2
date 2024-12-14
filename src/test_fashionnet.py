from models.fashionnet import FashionNet
from data_loading import get_data_loader
import logging

def test_fashionnet():
    """Test the FashionNet model's forward pass and shape compatibility."""
    logging.info("Initializing FashionNet model...")
    
    # create model instance with default parameters
    model = FashionNet(
        num_categories=50,  # as per DeepFashion dataset
        attr_group_dims=[7,3,3,4,6,3],  # attribute groups
        num_landmarks=8,
        pretrained=True
    )
    model = model.train()  # set to training mode
    
    # get a batch of data
    logging.info("Loading test batch from dataloader...")
    data_loader = get_data_loader('../data', 'train', batch_size=4)
    images, attributes, categories, landmarks = next(iter(data_loader))
    
    logging.info(f"Input shapes:")
    logging.info(f"- Images: {images.shape}")
    logging.info(f"- Attributes: {attributes.shape}")
    logging.info(f"- Categories: {categories.shape}")
    logging.info(f"- Landmarks: {landmarks.shape}")
    
    # forward pass
    logging.info("\nPerforming forward pass...")
    category_pred, attr_preds, landmark_pred = model(images)
    
    # verify output shapes
    logging.info("\nOutput shapes:")
    logging.info(f"- Category predictions: {category_pred.shape}")
    logging.info(f"- Landmark predictions: {landmark_pred.shape}")
    for i, attr_pred in enumerate(attr_preds):
        logging.info(f"- Attribute group {i} predictions: {attr_pred.shape}")
    
    # verify value ranges
    logging.info("\nChecking value ranges:")
    logging.info(f"- Category logits range: [{category_pred.min():.2f}, {category_pred.max():.2f}]")
    logging.info(f"- Landmark coordinates range: [{landmark_pred[:,:,:2].min():.2f}, {landmark_pred[:,:,:2].max():.2f}]")
    logging.info(f"- Landmark visibility range: [{landmark_pred[:,:,2].min():.2f}, {landmark_pred[:,:,2].max():.2f}]")
    
    # test memory efficiency
    logging.info("\nChecking memory usage...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"- Total parameters: {total_params:,}")
    logging.info(f"- Trainable parameters: {trainable_params:,}")
    
    # test gradient flow
    logging.info("\nTesting gradient flow...")
    try:
        # compute dummy loss
        loss = category_pred.mean() + sum(p.mean() for p in attr_preds) + landmark_pred.mean()
        loss.backward()
        
        # check if gradients are flowing
        has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        logging.info(f"- Gradients flowing properly: {has_grad}")
        
    except Exception as e:
        logging.error(f"Error during gradient flow test: {str(e)}")
    
    logging.info("\nTest completed successfully!")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    test_fashionnet() 