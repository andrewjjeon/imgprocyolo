from ultralytics import YOLO
import torch

'''
Validation on black and white images and color images separately
'''

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    img_size = 640
    epochs = 100
    batch_size = 16
    conf = 0.5
    iou = 0.5
    optimizer = 'auto'  # SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc.
    close_mosaic = 10  # Disables mosaic data augmentation in the last N (default 10) epochs to stabilize training before completion. Setting to 0 disables this feature.
    mosaic = 1

    weights_path = r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\runs\detect\train11\weights\best.pt"
    model = YOLO(weights_path)

    # Define the paths to the test sets
    color_test_path = r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\color\images\test"
    bw_test_path = r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\bw\images\test"

    model.val(data=r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\bw\bw_config.yaml",
            batch=batch_size, imgsz=img_size, patience=100, plots=True, 
            close_mosaic=close_mosaic, iou=iou, conf=conf, mosaic=mosaic, optimizer=optimizer)

    model.val(data=r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\color\color_config.yaml",
            batch=batch_size, imgsz=img_size, patience=100, plots=True, 
            close_mosaic=close_mosaic, iou=iou, conf=conf, mosaic=mosaic, optimizer=optimizer)
