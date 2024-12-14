import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class AttributeClassifier(nn.Module):
    """Simple attribute classifier using VGG16 backbone.
    
    Args:
        num_attributes (int): Number of attributes to predict
        pretrained (bool): Whether to use pretrained backbone weights
    """
    
    def __init__(self, num_attributes=26, pretrained=True):
        super(AttributeClassifier, self).__init__()
        
        # load vgg16 as backbone with proper weights initialization
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.vgg16(weights=weights)
        self.features = backbone.features  # conv layers
        
        # classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_attributes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize the weights of the non-pretrained layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Attribute logits of shape (batch_size, num_attributes)
        """
        # extract features using backbone
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        
        # get attribute predictions
        x = self.classifier(x)
        
        return x 