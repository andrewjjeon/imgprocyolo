# Image Processing for Fisheye Camera Image Object Detection

<h2>1. Introduction</h2>
    I participated in the 2024 AI City Challenge as part of a team from UW. Our team followed an ensemble learning approach where each person performed individual experiments and trained separate detectors. One person focused on finding additional fisheye camera image data, another focused on data augmentation, and I focused on image color transformation to improve detection. Specifically, a subset of the data was black&white images, and my job was to improve performance on them. I transformed my entire dataset to black&white images and trained yolov8 detectors on my transformed data. This resulted in performance improvements on the black&white validation data.

<h2>2. Image Transformation</h2>
    I followed a simple three step pipeline using OpenCV to transform my images to black & white.
    <ol>
    <li>Convert images to HSV</li>
    <li>Perform KMeans Clustering on HSV Images to split into colored vs black&white</li>
    <li>Transform Colored Images to Grayscale/Black&White</li>
    </ol>

<h2>3. Training and Evaluation of YOLO Detector</h2>
    I then use the Ultralytics package to load and train a yolov8m detector on the original dataset and my transformed dataset. I then trained and compared the validation set performance of the detector trained on the original data and the detector trained on my transformed data. There was a 10% improvement in mAP50-95 for all classes and a 6% improvement in mAP50 in model performance.

<div align="center">
    <img src="/media/yolo_results.png" width="700" />
    <p><em>Above: Performance of detector trained on original data. Below: Performance of detector trained on my transformed data.</em></p>
</div>

<div style="display: flex; justify-content: center; gap: 20px;">
    <div style="text-align: center;">
        <img src="/media/before1.jpg" width="400"/>
        <p><em>Detector Trained on Original Data Inference</em></p>
    </div>
    <div style="text-align: center;">
        <img src="/media/after1.jpg" width="400"/>
        <p><em>Detector Trained on Transformed Data Inference</em></p>
    </div>
</div>

<div style="display: flex; justify-content: center; gap: 20px;">
    <div style="text-align: center;">
        <img src="/images/yolo/before2.png" width="400"/>
        <p><em>Detector Trained on Original Data Inference</em></p>
    </div>
    <div style="text-align: center;">
        <img src="/images/yolo/after2.png" width="400"/>
        <p><em>Detector Trained on Transformed Data Inference</em></p>
    </div>
</div>

### Environment Setup

Please setup the environment as follows
```
conda create --name imgprocyolo python==3.8
conda activate imgprocyolo
conda install -c conda-forge opencv numpy tqdm
```

### Image Processing, Transformation and YOLOv8 Training, Validation
- Use bw_color_split.py to split and cluster your images into blackwhite vs color images
- Use transform.py to transform your color images to blackwhite for yolo retraining
- Use train_model to train however many yolo models you want
- Use validation.py and inference_1img.py to visualize metrics and inference results on an image.