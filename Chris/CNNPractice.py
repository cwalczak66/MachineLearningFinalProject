import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import matplotlib.pyplot as plt

TF_ENABLE_ONEDNN_OPTS=0

SIZE=128

import numpy as np
from PIL import Image


# Function to add random noise to the image
def add_random_noise(image, noise_factor=5):
    # Generate random noise (Gaussian noise with tiny variation)
    noise = np.random.normal(0, noise_factor, image.shape)
    noisy_image = image + noise

    # Clip to ensure pixel values remain between 0 and 1
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image

def show_image(image, title="Image"):
    if len(image.shape) == 3 and image.shape[-1] == 1:  # Grayscale (128, 128, 1)
        image = image.squeeze()  # Remove the last dimension -> (128, 128)

    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def random_zoom(Xset, zoom_range=(0.8, 1.2)):
    """
    Randomly zooms images in X_train using Pillow.

    Args:
    - X_train (numpy array): Array of shape (4000, 128, 128)
    - zoom_range (tuple): Range for zoom scaling (min_zoom, max_zoom)

    Returns:
    - numpy array: Transformed array with randomly zoomed images
    """
    Xset = np.squeeze(Xset, axis=-1)  # Removes the last axis

    num_images, height, width = Xset.shape
    zoomed_images = np.zeros_like(Xset)

    for i in range(num_images):
        img = Image.fromarray(Xset[i])
        zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])

        # Compute new size
        new_h = int(height * zoom_factor)
        new_w = int(width * zoom_factor)

        # Resize
        zoomed_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Crop or pad to 128x128
        if zoom_factor > 1.0:  # Zoom in (crop)
            left = (new_w - width) // 2
            top = (new_h - height) // 2
            zoomed_img = zoomed_img.crop((left, top, left + width, top + height))
        else:  # Zoom out (pad)
            result = Image.new("L", (width, height), color=1)
            paste_x = (width - new_w) // 2
            paste_y = (height - new_h) // 2
            result.paste(zoomed_img, (paste_x, paste_y))
            zoomed_img = result

        # Convert back to numpy
        zoomed_images[i] = np.array(zoomed_img)

    Xset = np.expand_dims(Xset, axis=-1)
    return zoomed_images


xset = np.load("Xset.npy")

print(xset.shape)
yset = np.load("Yset.npy")



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 1)))
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
# X_shuffled = X_shuffled / 255.0  # Scale pixel values to [0,1]

# Reshape if grayscale
X_shuffled = X_shuffled.reshape(-1, SIZE, SIZE, 1)  # Use (4000, 128, 128, 3) if RGB

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_shuffled, Y_shuffled, test_size=0.3, random_state=42)
show_image(X_train[0])
X_train = random_zoom(X_train, zoom_range=(0.9, 1.1))
X_train = np.array([add_random_noise(img) for img in X_train])

X_test = random_zoom(X_test, zoom_range=(0.9, 1.1))
X_test = np.array([add_random_noise(img) for img in X_test])

X_train = X_train/255.0
X_test = X_test/255.0
#
#
#
# # Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
#
#
#
# # Train the model using the augmented data from the ImageDataGenerator
# #model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))
#
#
model.save('lego_piece_classifier.h5')