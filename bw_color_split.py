import os.path as osp
import cv2
import glob
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import shutil
import os
from sklearn.model_selection import train_test_split

# histograms = []
train_image_dir = r'C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\images\train'
test_image_dir = r'C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\images\test'
train_label_dir = r'C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\labels\train'
test_label_dir = r'C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\labels\test'

# Load train and test image paths
train_images = glob.glob(os.path.join(train_image_dir, '*'))
test_images = glob.glob(os.path.join(test_image_dir, '*'))

# Load train and test label paths
train_labels = glob.glob(os.path.join(train_label_dir, '*'))
test_labels = glob.glob(os.path.join(test_label_dir, '*'))

# Calculate saturation of each pixel
pixel_saturation_train = []
for img_path in tqdm(train_images):
    image = cv2.imread(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()  # Calculate mean saturation across all pixels
    pixel_saturation_train.append(saturation)

pixel_saturation_test = []
for img_path in tqdm(test_images):
    image = cv2.imread(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()  # Calculate mean saturation across all pixels
    pixel_saturation_test.append(saturation)

# Reshape data for clustering
X_train = np.array(pixel_saturation_train).reshape(-1, 1)
X_test = np.array(pixel_saturation_test).reshape(-1, 1)

# Perform K-means clustering for train data
np.random.seed(42)  # Set a random seed for reproducibility
kmeans_train = KMeans(n_clusters=2)
kmeans_train.fit(X_train)
train_cluster_labels = kmeans_train.labels_

# Perform K-means clustering for test data
np.random.seed(42)  # Set a random seed for reproducibility
kmeans_test = KMeans(n_clusters=2)
kmeans_test.fit(X_test)
test_cluster_labels = kmeans_test.labels_

# Visualization
# Display 5 images from each cluster
for cluster_label in range(2):
    train_cluster_images = [img_path for img_path, label in zip(train_images, train_cluster_labels) if
                            label == cluster_label]
    random_train_cluster_images = random.sample(train_cluster_images, min(3, len(train_cluster_images)))
    cluster_type = "Train BW" if cluster_label == 0 else "Train Color"

    print(f"Cluster {cluster_label + 1} ({cluster_type}):")
    for img_path in random_train_cluster_images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(cluster_type)
        plt.imshow(img)
        plt.axis("off")
        plt.show()

# Display 5 images from each cluster
for cluster_label in range(2):
    test_cluster_images = [img_path for img_path, label in zip(test_images, test_cluster_labels) if
                           label == cluster_label]
    random_test_cluster_images = random.sample(test_cluster_images, min(3, len(test_cluster_images)))
    cluster_type = "Test Color" if cluster_label == 0 else "Test BW"

    print(f"Cluster {cluster_label + 1} ({cluster_type}):")
    for img_path in random_test_cluster_images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(cluster_type)
        plt.imshow(img)
        plt.axis("off")
        plt.show()

# Move train images to appropriate directories based on clustering results
for img_path, cluster_label in zip(train_images, train_cluster_labels):
    cluster_type = "BlackWhite" if cluster_label == 0 else "Colored"
    new_dir = os.path.join(train_image_dir, cluster_type)
    os.makedirs(new_dir, exist_ok=True)
    shutil.move(img_path, new_dir)

# Move test images to appropriate directories based on clustering results
for img_path, cluster_label in zip(test_images, test_cluster_labels):
    cluster_type = "Colored" if cluster_label == 0 else "BlackWhite"
    new_dir = os.path.join(test_image_dir, cluster_type)
    os.makedirs(new_dir, exist_ok=True)
    shutil.move(img_path, new_dir)

# Move train labels to appropriate directories based on clustering results
for label_path, cluster_label in zip(train_labels, train_cluster_labels):
    cluster_type = "BlackWhite" if cluster_label == 0 else "Colored"
    new_dir = os.path.join(train_label_dir, cluster_type)
    os.makedirs(new_dir, exist_ok=True)
    shutil.move(label_path, new_dir)

# Move test labels to appropriate directories based on clustering results
for label_path, cluster_label in zip(test_labels, test_cluster_labels):
    cluster_type = "Colored" if cluster_label == 0 else "BlackWhite"
    new_dir = os.path.join(test_label_dir, cluster_type)
    os.makedirs(new_dir, exist_ok=True)
    shutil.move(label_path, new_dir)