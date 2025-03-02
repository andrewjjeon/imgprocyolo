import cv2
import glob
import os
from tqdm import tqdm

def load_image_paths(directory):
    return glob.glob(os.path.join(directory, "*.jpg"))

def transform2bw(image_paths, output_dir):
    """transform colored images to blackwhite"""
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in tqdm(image_paths, desc="transforming to bw"):
        image = cv2.imread(img_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, gray_image)

if __name__ == "__main__":
    img_dir = r'C:/Users/Andrew Jeon/OneDrive/Desktop/Fisheye/images'
    
    for dataset in ["train", "test"]:
        color_dir = os.path.join(img_dir, dataset, "color")
        bw_output_dir = os.path.join(img_dir, dataset, "bw")
        
        image_paths = load_image_paths(color_dir)
        transform2bw(image_paths, bw_output_dir)
