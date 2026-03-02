"""
Data preprocessing module for MNIST dataset (PyTorch version)
"""
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data(batch_size=128):
    """
    Load MNIST dataset and create data loaders
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1] and adds channel dimension
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    
    return train_loader, val_loader, test_loader

def get_sample_images(loader, num_samples=10):
    """
    Get sample images from a data loader for visualization
    """
    data_iter = iter(loader)
    images, labels = next(data_iter)
    return images[:num_samples], labels[:num_samples]

def visualize_samples(images, labels, num_samples=10):
    """
    Visualize sample images
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i].squeeze().numpy()  # Remove channel dimension
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()

def prepare_for_mlp(images):
    """
    Flatten images for MLP
    """
    return images.view(images.size(0), -1)

if __name__ == "__main__":
    # Test the preprocessing
    train_loader, val_loader, test_loader = load_and_preprocess_data()
    images, labels = get_sample_images(train_loader)
    visualize_samples(images, labels)