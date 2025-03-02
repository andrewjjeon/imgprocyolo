import os
import cv2
import glob
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans

img_dir = r'C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\images'
label_dir = r'C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\labels'

train_img_dir = f"{img_dir}/train"
test_img_dir = f"{img_dir}/test"
train_label_dir = f"{label_dir}/train"
test_label_dir = f"{label_dir}/test"

def load_image_paths(directory):
    return glob.glob(os.path.join(directory, '*'))

def compute_saturation(image_paths):
    """Compute mean saturation for each image."""
    saturations = []
    for img_path in tqdm(image_paths, desc="computing mean saturation"):
        image = cv2.imread(img_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturations.append(hsv[:, :, 1].mean())
    return np.array(saturations).reshape(-1, 1)

def perform_clustering(X, n_clusters=2):
    """KMeans Clustering on images."""
    np.random.seed(42)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    return kmeans.fit_predict(X)

def visualize_clusters(image_paths, cluster_labels, cluster_names, num_samples=3):
    """Visualize 3 random images from each cluster to confirm correct clustering."""
    for label, cluster_name in cluster_names.items():
        cluster_images = [img for img, lbl in zip(image_paths, cluster_labels) if lbl == label]
        sampled_images = random.sample(cluster_images, min(num_samples, len(cluster_images)))
        
        print(f"Cluster {label + 1} ({cluster_name}):")
        for img_path in sampled_images:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.title(cluster_name)
            plt.imshow(img)
            plt.axis("off")
            plt.show()

def move_files(file_paths, cluster_labels, base_dir, cluster_names):
    """Move files into categorized directories based on clustering labels."""
    for file_path, label in zip(file_paths, cluster_labels):
        dest_dir = os.path.join(base_dir, cluster_names[label])
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(file_path, dest_dir)

# Load images and labels
train_images = load_image_paths(train_img_dir)
test_images = load_image_paths(test_img_dir)
train_labels = load_image_paths(train_label_dir)
test_labels = load_image_paths(test_label_dir)

# Compute mean saturation values
X_train = compute_saturation(train_images)
X_test = compute_saturation(test_images)

# Perform clustering
train_cluster_labels = perform_clustering(X_train)
test_cluster_labels = perform_clustering(X_test)

# Define cluster names
cluster_mapping = {0: "bw", 1: "color"}

# Visualize results
visualize_clusters(train_images, train_cluster_labels, cluster_mapping)
visualize_clusters(test_images, test_cluster_labels, cluster_mapping)

# Move files into corresponding clusters
move_files(train_images, train_cluster_labels, train_img_dir, cluster_mapping)
move_files(test_images, test_cluster_labels, test_img_dir, cluster_mapping)
move_files(train_labels, train_cluster_labels, train_label_dir, cluster_mapping)
move_files(test_labels, test_cluster_labels, test_label_dir, cluster_mapping)