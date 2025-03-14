# LEGO Image Classification with Softmax Regression

## Overview
This project implements a softmax regression model using stochastic gradient descent (SGD) to classify LEGO images. The dataset consists of images and one-hot encoded labels stored in NumPy files.

## Features
- Loads and normalizes an image dataset
- Implements softmax regression with mini-batch SGD
- Includes L2 regularization for better generalization
- Evaluates model performance using accuracy metrics
- Visualizes trained weight vectors as images

## Requirements
This project requires Python and the following dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

## Dataset
The dataset consists of two files:
- `Xset.npy`: Contains the image data, normalized between 0 and 1.
- `Yset.npy`: Contains one-hot encoded labels.

### Parameters
- `epsilon`: Learning rate (default: 0.1)
- `batchSize`: Mini-batch size for SGD (default: 100)
- `alpha`: L2 regularization parameter (default: 0.1)

## Training Process
1. Loads and normalizes dataset
2. Splits dataset into training (70%) and testing (30%)
3. Applies softmax regression with mini-batch SGD
4. Evaluates accuracy on the test set

## Model Output
- Prints training loss over given epochs
- Displays final test accuracy
- Optionally visualizes weight vectors for each class

## Results
- Accuracy: ~80%

