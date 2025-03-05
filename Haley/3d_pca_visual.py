import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

## NOTE PLEASE RUN 2D NOTEBOOK FIRST TO GENERATE NEEDED FILES###################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load data
x_train = np.load("Haley/x_train_subset.npy")
y_train = np.load("Haley/y_train_subset.npy")

lego_Y_labels = [3001, 3003, 3023, 3794, 4150, 4286, 6632, 18654, 43093, 54200]

# Flatten each image into a 1D vector
x_train_flat = x_train.reshape(x_train.shape[0], -1)  # (num_samples, flattened_features)

# Apply PCA (reduce to 3D)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(x_train_flat)

# Convert one-hot labels back to class indices
labels = np.argmax(y_train, axis=1)

# Map indices to LEGO brick labels
lego_labels = [lego_Y_labels[idx] for idx in labels]

# Define color palette for visualization
unique_classes = np.unique(labels)
palette = sns.color_palette("hsv", len(unique_classes))
color_map = {cls: palette[i] for i, cls in enumerate(unique_classes)}

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D space with mapped colors
scatter = ax.scatter(
    pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
    c=[color_map[lbl] for lbl in labels], alpha=0.7
)

# Labels and Title
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.set_title("3D PCA Visualization of LEGO Brick Images")

# Create a manual legend mapping LEGO part numbers
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cls], markersize=8) 
           for cls in unique_classes]
ax.legend(handles, [lego_Y_labels[cls] for cls in unique_classes], title="LEGO Types", loc="upper right")

plt.show()


# EMBEDDINGS VISUAL ###############################################################################################
embeddings = np.load("Haley/embeddings.npy")
embedding_labels = np.load("Haley/embedding_labels.npy")


label_names = [lego_Y_labels[label] for label in embedding_labels]



# Apply PCA (reduce to 3D)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeddings)  # Use your extracted embeddings

# Create 3D plot
# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(
    pca_result[:, 0], 
    pca_result[:, 1], 
    pca_result[:, 2], 
    c=embedding_labels,  
    cmap='tab10',  
    alpha=0.7
)

# Labels and title
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.set_title("3D PCA Visualization of Last-Layer Embeddings")

# Generate legend patches for unique LEGO names
unique_labels = np.unique(embedding_labels)
legend_patches = [mpatches.Patch(color=plt.cm.tab10(i / len(unique_labels)), label=lego_Y_labels[i]) 
                  for i in unique_labels]

# Move legend outside the plot
ax.legend(handles=legend_patches, title="LEGO Types", loc="center left", bbox_to_anchor=(1.05, 0.5))

plt.show()
plt.close()
