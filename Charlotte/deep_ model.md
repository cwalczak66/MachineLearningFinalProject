# Deep Model: Multi-Layer Perceptron (MLP) Model

This model is a Multi-Layer Perceptron (MLP) designed for LEGO image classification. It uses a fully connected feedforward network with dropout and batch normalization for better generalization.

###  Model Architecture
- Input Layer: Accepts flattened image vectors.
- Hidden Layers:
  - 512 neurons (ReLU activation, batch normalization, dropout 0.3).
  - 256 neurons (ReLU activation, batch normalization, dropout 0.3).
- Output Layer: Uses softmax activation for multi-class classification.

### 3. Training Setup
- Optimizer: Adam (learning_rate = 0.001).
- Loss Function: Categorical Crossentropy.
- Batch Size: 64.
- Epochs: 100.
- Early Stopping: Stops training if validation loss doesn’t improve for 10 epochs.

### 4. Model Evaluation
- Splits dataset into 70% training, 30% testing.
- Logs training and validation loss.
- Plots the loss curve to visualize performance.
- Saves the trained model as `lego_classifier_model.h5`.

## Performance
- Latest Accuracy: 88.83%

