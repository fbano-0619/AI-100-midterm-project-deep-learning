"""
Multi-Layer Perceptron (MLP) model for MNIST classification (PyTorch version)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for MNIST classification
    
    Architecture:
    - Input: 784 features (28x28 flattened)
    - Hidden Layer 1: 512 neurons with ReLU + Dropout
    - Hidden Layer 2: 256 neurons with ReLU + Dropout
    - Hidden Layer 3: 128 neurons with ReLU + Dropout
    - Output: 10 classes
    """
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout_rate=0.3):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer to first hidden
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        return x

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model creation
    model = MLP()
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    test_input = torch.randn(5, 1, 28, 28)  # Batch of 5 images
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")