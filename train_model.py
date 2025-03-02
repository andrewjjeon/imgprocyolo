from ultralytics import YOLO
import torch.optim as optim
import torch
import os


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

    model = YOLO("yolov8m.pt")  # build a new model from scratch
    model.train(data=r"C:\Users\Andrew Jeon\OneDrive\Desktop\Fisheye\bw_transformed\images\transformed_bw_config.yaml",
                epochs=epochs, patience=100, plots=True, batch=batch_size, imgsz=img_size, close_mosaic=close_mosaic,
                iou=iou, conf=conf, mosaic=mosaic, optimizer=optimizer)  # train the model








