import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_lego_dataset(xset_file, yset_file):

    # Load dataset
    if not os.path.exists(xset_file) or not os.path.exists(yset_file):
        print("\nError: Xset or Yset file is missing\n")
        exit(1)
    
    # Normalize images
    images = np.load(xset_file) / 255.0  
    # Labels already one-hot encoded
    labels = np.load(yset_file)  

    ''' Dataset descriptor logs for debugging
    print("\nDataset Loaded Successfully")
    print(f"Total Images: {images.shape[0]}, Features per Image: {images.shape[1]}")
    print(f"Total Labels: {labels.shape[0]}")
    print(f"Yset.npy shape: {labels.shape}")
    '''

    return images, labels 


# Stochastic gradient descent (SGD) to optimize the weight matrix Wtilde
def softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):
    epochs = 100
    num_samples, num_features = trainingImages.shape
    num_classes = trainingLabels.shape[1]  

    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = np.hstack([trainingImages, np.ones((num_samples, 1))])
    testingImages = np.hstack([testingImages, np.ones((testingImages.shape[0], 1))])

    # Initialize weights with small random values
    W = np.random.randn(num_features + 1, num_classes) * 1e-5

    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        trainingImages, trainingLabels = trainingImages[indices], trainingLabels[indices]

        # Mini-batch SGD
        for i in range(0, num_samples, batchSize):
            X_batch = trainingImages[i:i+batchSize]
            y_batch = trainingLabels[i:i+batchSize]

            if X_batch.shape[0] == 0:
                continue  

            # Ensure correct shape for y_batch
            y_batch = y_batch.squeeze()

            # Compute softmax probabilities
            scores = X_batch @ W
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Compute loss
            loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-9), axis=1))
            loss += (alpha / (2 * num_samples)) * np.sum(W[:-1, :] ** 2)

            # Compute gradient
            gradient = (X_batch.T @ (probs - y_batch)) / batchSize + (alpha / num_samples) * np.vstack([W[:-1, :], np.zeros((1, num_classes))])

            # Update weights
            W -= epsilon * gradient

        # Log the last 20 epochs
        if epoch >= epochs - 20:
            print(f"Epoch {epoch + 1} loss = {loss:.4f}")

    # Test set accuracy calculation
    test_scores = testingImages @ W
    test_preds = np.argmax(test_scores, axis=1)
    test_labels = np.argmax(testingLabels, axis=1)
    accuracy = np.mean(test_preds == test_labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    return W


if __name__ == "__main__":
    # Load data
    xset_file = "Xset.npy"
    yset_file = "Yset.npy"

    # loading images
    images, labels = load_lego_dataset(xset_file, yset_file)

    if images.size == 0 or labels.size == 0:
        print("Error: Dataset is empty\n")
        exit(1)

    # Flatten images
    images = images.reshape(images.shape[0], -1)

    # Ensure correct class count
    num_classes = labels.shape[1]

    # Split into training and testing sets
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.3, random_state=42)

    ''' Testing and Training Shape prints for debugging
    print(f"Training Images Shape: {images_train.shape}")
    print(f"Training Labels Shape: {labels_train.shape}")
    print(f"Testing Images Shape: {images_test.shape}")
    print(f"Testing Labels Shape: {labels_test.shape}")
    '''

    # Model training
    Wtilde = softmaxRegression(images_train, labels_train, images_test, labels_test, epsilon=0.1, batchSize=100, alpha=0.1)

    # Visualizing weight vectors for each class (optional)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < num_classes:
            ax.imshow(Wtilde[:-1, i].reshape(128, 128), cmap='gray')
            ax.set_title(f"Class {i}", fontsize=8)
        ax.axis('off')
    # plt.show()
