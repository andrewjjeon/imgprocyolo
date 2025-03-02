from ultralytics import YOLO

# Load the trained model
model = YOLO(r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\runs\detect\train11\weights\best.pt")
bw_image_path = r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\images\train\camera3_N_0.png"
colored_image_path = r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\images\train\camera3_A_399.png"

# Run inference on image
results_bw = model(bw_image_path, save=True)  # BB, segmentation, mask, etc.

results_colored = model(colored_image_path, save=True)

