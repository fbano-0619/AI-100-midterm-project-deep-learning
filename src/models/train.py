"""
Training script for both MLP and CNN models (PyTorch version)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import load_and_preprocess_data, get_sample_images, visualize_samples
from mlp_model import MLP, count_parameters as count_mlp_params
from cnn_model import CNN, count_parameters as count_cnn_params

class ModelTrainer:
    def __init__(self, device=None):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load data
        print("\nLoading and preprocessing data...")
        self.train_loader, self.val_loader, self.test_loader = load_and_preprocess_data(batch_size=128)
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        os.makedirs('models/saved', exist_ok=True)
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, model, val_loader, criterion):
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, model, model_name, epochs=15, lr=0.001):
        """Train a model"""
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        print(f"Model parameters: {count_mlp_params(model):,}" if model_name == "MLP" else f"Model parameters: {count_cnn_params(model):,}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(model, self.train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(model, self.val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'models/saved/{model_name.lower()}_best.pt')
                print(f"  → Best model saved! (Val Acc: {val_acc:.4f})")
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(f'models/saved/{model_name.lower()}_best.pt'))
        
        # Final test evaluation
        test_loss, test_acc = self.validate(model, self.test_loader, criterion)
        print(f"\n{model_name} Test Accuracy: {test_acc:.4f}")
        print(f"{model_name} Test Loss: {test_loss:.4f}")
        
        return model, history, test_acc
    
    def plot_training_history(self, mlp_history, cnn_history):
        """Plot training history for both models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # MLP Accuracy
        axes[0, 0].plot(mlp_history['train_acc'], label='Training', marker='o')
        axes[0, 0].plot(mlp_history['val_acc'], label='Validation', marker='s')
        axes[0, 0].set_title('MLP Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_ylim([0.9, 1.0])
        
        # MLP Loss
        axes[0, 1].plot(mlp_history['train_loss'], label='Training', marker='o')
        axes[0, 1].plot(mlp_history['val_loss'], label='Validation', marker='s')
        axes[0, 1].set_title('MLP Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # CNN Accuracy
        axes[1, 0].plot(cnn_history['train_acc'], label='Training', marker='o')
        axes[1, 0].plot(cnn_history['val_acc'], label='Validation', marker='s')
        axes[1, 0].set_title('CNN Model Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0.9, 1.0])
        
        # CNN Loss
        axes[1, 1].plot(cnn_history['train_loss'], label='Training', marker='o')
        axes[1, 1].plot(cnn_history['val_loss'], label='Validation', marker='s')
        axes[1, 1].set_title('CNN Model Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train MLP
    mlp_model = MLP()
    mlp_model, mlp_history, mlp_acc = trainer.train_model(mlp_model, "MLP", epochs=15)
    
    # Train CNN
    cnn_model = CNN()
    cnn_model, cnn_history, cnn_acc = trainer.train_model(cnn_model, "CNN", epochs=15)
    
    # Plot results
    trainer.plot_training_history(mlp_history, cnn_history)
    
    # Print comparison
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    print(f"MLP Test Accuracy: {mlp_acc:.4f}")
    print(f"CNN Test Accuracy: {cnn_acc:.4f}")
    print(f"Improvement with CNN: {((cnn_acc - mlp_acc) * 100):.2f}%")

if __name__ == "__main__":
    main()