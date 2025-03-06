import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
TF_ENABLE_ONEDNN_OPTS=0
xset = np.load("Xset.npy")
xset = xset/255.0
print(xset.shape)
yset = np.load("Yset.npy")
#
# img1 = xset[0,:,:]
# for i in range(1,len(img1)):
#     print(img1[i])
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


X = np.load("Xset.npy")
Y = np.load("Yset.npy")

indices = np.random.permutation(len(X))

# Shuffle X and y using the same random order
X_shuffled = X[indices]
Y_shuffled = Y[indices]

# Normalize images
X_shuffled = X_shuffled / 255.0  # Scale pixel values to [0,1]

# Reshape if grayscale
X_shuffled = X_shuffled.reshape(-1, 128, 128, 1)  # Use (4000, 128, 128, 3) if RGB

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_shuffled, Y_shuffled, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))