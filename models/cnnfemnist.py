import torch.nn as nn
import torch.nn.functional as F

class CNNFEMNIST(nn.Module):
    """
    A standard Convolutional Neural Network (CNN) for the FEMNIST dataset.
    This architecture is commonly used in federated learning research papers.
    
    The model expects input images of size 1x28x28 (channels x height x width).
    """
    def __init__(self):
        super(CNNFEMNIST, self).__init__()
        # First convolutional block
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=1)
        # After conv1: 32x26x26
        # After relu
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool1: 32x13x13

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1)
        # After conv2: 64x11x11
        # After relu
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool2: 64x5x5
        
        # Flatten the feature map
        # Flattened size: 64 * 5 * 5 = 1600
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.fc2 = nn.Linear(2048, 62) # 62 classes for FEMNIST

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Logit scores for each class, shape (batch_size, 62)
        """
        # Apply first convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Apply second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten the tensor for the fully connected layers
        # The view function reshapes the tensor. -1 infers the batch size.
        x = x.view(-1, 64 * 5 * 5)
        
        # Apply fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        
        # Output layer
        x = self.fc2(x)
        
        # The CrossEntropyLoss function in PyTorch applies log_softmax internally,
        # so we return the raw logits.
        return x
    

def get_model():
    return CNNFEMNIST()