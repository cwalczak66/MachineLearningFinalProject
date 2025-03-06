import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define TensorFlow components as variables
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Load and preprocess dataset
def load_lego_dataset(xset_file, yset_file):
    if not os.path.exists(xset_file) or not os.path.exists(yset_file):
        print("\nError: Xset or Yset file is missing\n")
        exit(1)
    
    # Normalize
    images = np.load(xset_file) / 255.0  
    # One-hot encoded labels
    labels = np.load(yset_file)  
    return images, labels

# Load dataset
xset_file = "Xset.npy"
yset_file = "Yset.npy"
images, labels = load_lego_dataset(xset_file, yset_file)

if images.size == 0 or labels.size == 0:
    print("Error: Dataset is empty\n")
    exit(1)

# Flatten images for MLP
images = images.reshape(images.shape[0], -1)

# Train-test split (70% train, 30% test)
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Define improved model (Multi-Layer Perceptron)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(images.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(labels.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(images_train, labels_train, 
                    epochs=100, batch_size=64,  
                    validation_data=(images_test, labels_test),
                    callbacks=[early_stopping], verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(images_test, labels_test, verbose=2)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save("lego_classifier_model.h5")

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.show()
