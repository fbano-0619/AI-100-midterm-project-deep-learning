"""
Visualization utilities for model evaluation (PyTorch version)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def plot_confusion_matrix(model, test_loader, class_names, device='cpu', title="Confusion Matrix"):
    """
    Plot confusion matrix for model predictions
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return cm

def plot_misclassified_examples(model, test_loader, class_names, device='cpu', num_examples=10):
    """
    Plot examples of misclassified images
    """
    model.eval()
    model.to(device)
    
    misclassified_images = []
    misclassified_true = []
    misclassified_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Find misclassified
            misclassified_mask = (predicted.cpu() != labels)
            
            if misclassified_mask.any():
                misclassified_images.extend(images[misclassified_mask].cpu())
                misclassified_true.extend(labels[misclassified_mask])
                misclassified_pred.extend(predicted.cpu()[misclassified_mask])
            
            if len(misclassified_images) >= num_examples:
                break
    
    if len(misclassified_images) == 0:
        print("No misclassified examples found!")
        return
    
    # Plot examples
    num_examples = min(num_examples, len(misclassified_images))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_examples):
        img = misclassified_images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {class_names[misclassified_true[i]]}\nPred: {class_names[misclassified_pred[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/misclassified_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_classification_report(model, test_loader, class_names, device='cpu'):
    """
    Generate and print classification report
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Generate report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    print("\nClassification Report:")
    print("="*50)
    print(report)
    
    # Save to file
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    
    return report

if __name__ == "__main__":
    # Test with a sample model (if available)
    try:
        from cnn_model import CNN
        from data_preprocessing import load_and_preprocess_data
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        _, _, test_loader = load_and_preprocess_data(batch_size=128)
        
        # Load model
        model = CNN()
        model.load_state_dict(torch.load('models/saved/cnn_best.pt', map_location=device))
        
        class_names = [str(i) for i in range(10)]
        plot_confusion_matrix(model, test_loader, class_names, device)
        generate_classification_report(model, test_loader, class_names, device)
        plot_misclassified_examples(model, test_loader, class_names, device)
    except Exception as e:
        print(f"Error: {e}")
        print("No saved model found. Run train.py first.")