import cv2
import glob
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import shutil
import os
import os.path as osp
import cv2
import pickle

test_image_dir = r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\test"

# Load test image path
test_images = glob.glob(os.path.join(test_image_dir, '*'))

# Extract features and perform clustering for each image
histograms = []
for img_path in tqdm(test_images):
    # Extract features
    image = cv2.imread(img_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256]).flatten()
    # hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256]).flatten()
    # hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256]).flatten()
    # hist = np.hstack((hist_r, hist_g, hist_b))
    # histograms.append(hist)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256]).flatten()  # Calculate histogram
    histograms.append(hist)

histograms = np.array(histograms)

# Standardize the features
scaler = StandardScaler()
histograms_scaled = scaler.fit_transform(histograms)

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(histograms_scaled)

# Print information for debugging
print("Number of images:", len(test_images))
print("Shape of histograms:", np.array(histograms).shape)
print("Shape of histograms_scaled:", histograms_scaled.shape)
print("Shape of cluster_labels:", cluster_labels.shape)

# Separate images into black & white and colored based on cluster labels
bw_images = [img for i, img in enumerate(test_images) if cluster_labels[i] == 1]
color_images = [img for i, img in enumerate(test_images) if cluster_labels[i] == 0]

print("Number of black & white images:", len(bw_images))
print("Number of colored images:", len(color_images))

# Visualize three black & white images
plt.figure(figsize=(10, 6))
plt.suptitle('Black & White Images')
for i, img_path in enumerate(random.sample(bw_images, 3)):
    img = cv2.imread(img_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
plt.show()

# Visualize three colored images
plt.figure(figsize=(10, 6))
plt.suptitle('Colored Images')
for i, img_path in enumerate(random.sample(color_images, 3)):
    img = cv2.imread(img_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
plt.show()
