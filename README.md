# ml-car-count
# Drive-Thru Car Counting with YOLOv8

This project uses the YOLOv8 object detection model to count cars in a drive-thru video by tracking their dwell time within a defined Region of Interest (ROI). It processes video input, detects cars, calculates Intersection over Union (IOU) with the ROI, and outputs an annotated video with car counts.

## Features
- **Object Detection**: Uses YOLOv8 (medium model, `yolov8m.pt`) to detect cars.
- **ROI Tracking**: Counts cars that stay within a specified ROI for at least 5 seconds.
- **Visualization**: Draws bounding boxes (green if IOU > 0.3, red otherwise) and displays car IDs, IOU values, and total count.
- **Output**: Saves an annotated video with processing metrics.

## Prerequisites
- **Python 3.7+**: Ensure Python is installed.
- **Google Colab**: Recommended for free GPU access (T4), though it can run locally with a compatible GPU.
- **Dependencies**:
  - `opencv-python` (cv2)
  - `numpy`
  - `ultralytics` (for YOLOv8)
  - `torch` (automatically installed with ultralytics)
