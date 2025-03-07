from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

class_mapping = {
    0: 3001,
    1: 3003,
    2: 3023,
    3: 3794,
    4: 4150,
    5: 4286,
    6: 6632,
    7: 18654,
    8: 43093,
    9: 54200
}

def show_image(image, title="Image"):
    if len(image.shape) == 3 and image.shape[-1] == 1:  # Grayscale (128, 128, 1)
        image = image.squeeze()  # Remove the last dimension -> (128, 128)

    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def loop_through_pixels(img_array):
    height, width = img_array.shape[:2]
    for y in range(height):
        for x in range(width):
            pixel = img_array[y, x]
            if pixel >= 200:
                img_array[y, x] = 0

    show_image(img_array)
    return img_array
            # You can do something with the pixel value here

# Load the saved model
model = load_model('C:/Users/chris/WPI/Machine Learning/FinalProject/Chris/lego_piece_classifier.h5')


# Load the image and convert it to grayscale (128x128, 1 channel)
img_path = 'C:/Users/chris/WPI/Machine Learning/FinalProject/Chris/PreformanceMeasures/olala2.jpg'
img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')

# Convert image to array and scale pixel values
img_array = image.img_to_array(img)

img_array = loop_through_pixels(img_array)

img_array = img_array/255

# Add batch dimension (model expects shape: (1, 128, 128, 1))
img_array = np.expand_dims(img_array, axis=0)


print(img_array)  # Should print (1, 128, 128, 1)



# Predict class probabilities
predictions = model.predict(img_array)

print(predictions)

# Get the class with the highest probability
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

print(f'Predicted LEGO piece class: {class_mapping[predicted_class]}, Confidence: {confidence:.2f}')
