import numpy as np
from matplotlib import pyplot as plt

#one hot map: [3001,3003,3023,3794,4150,4286,6632,18654,43093,54200]
xtrain = np.load("Xset.npy")
ytrain = np.load("Yset.npy")

n = 7999

print(ytrain[n])

image = xtrain[n]
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
