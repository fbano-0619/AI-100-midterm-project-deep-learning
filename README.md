# Deep Learning Classification Project: MNIST Handwritten Digit Recognition

## Project Overview
This project implements multiple deep learning models to classify handwritten digits from the MNIST dataset. We compare the performance of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures.

## Team Members
- Robert Fan

## Dataset
- **Source**: MNIST dataset (70,000 grayscale images of handwritten digits 0-9)
- **Split**: 54,000 training images, 6,000 validation images, 10,000 testing images
- **Preprocessing**: Normalization (mean=0.1307, std=0.3081), reshaping, and tensor conversion

## Models Implemented
1. **Multi-Layer Perceptron (MLP)**: 3-layer fully connected network
2. **Convolutional Neural Network (CNN)**: 2 convolutional layers + 1 dense layer with dropout

## Requirements
See `requirements.txt` for detailed dependencies.

## Results
- MLP Accuracy: **98.03%**
- CNN Accuracy: **99.50%**
- Improvement with CNN: **1.47%**

## How to Run
```bash
pip install -r requirements.txt
python src/models/train.py