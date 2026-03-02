"""
Convolutional Neural Network (CNN) model for MNIST classification (PyTorch version)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST classification
    
    Architecture:
    - Conv Block 1: Conv2d(1→32, 3x3) + Conv2d(32→32, 3x3) + MaxPool + Dropout
    - Conv Block 2: Conv2d(32→64, 3x3) + Conv2d(64→64, 3x3) + MaxPool + Dropout
    - Dense Layers: 128 neurons + Dropout → 10 output classes
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Calculate size after convolutions
        # Input: 28x28 -> After block1: 14x14 -> After block2: 7x7
        # Final channels: 64, so flattened size = 64 * 7 * 7 = 3136
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model creation
    model = CNN()
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    test_input = torch.randn(5, 1, 28, 28)  # Batch of 5 images
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")