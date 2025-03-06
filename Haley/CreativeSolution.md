# Unique Solution

## Method - Feature Engineering
Local Binary Patterns
- Method for extracting texture features from an image. 
- LBP works by comparing the intensity of each pixel with its neighbors and assigning a binary value based on whether the neighboring pixel is brighter or darker than the central pixel
- This process is repeated for every pixel in the image to generate a pattern that captures local texture.

CNN Features
- CNN features extract high-level features from input images.
- One version of implementation uses a pretrained MobileNetV2 model to extract features from the images.
- The other version takes the pretrained MobileNetV2 and trains it on the LEGO dataset, then those embeddings are extracted.


Combining LBP with CNN Features
- Concatenate the texture features from the LBP with the feature form the CNN.
- The combination of low-level (texture) and high-level (CNN) features leads to more comprehensive representations of the image, helping the model generalize better.

Training Classifier
- Train a Random Forest classifier on the combined features.
- Good for high-dimensional data
- Robust to noise and outliers

