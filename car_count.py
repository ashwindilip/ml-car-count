import cv2
import numpy as np
import time
from ultralytics import YOLO

# Not necessary on Colab, but if you choose to do visualization during runtime itself (better option when running locally) - then this function is convenient
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def calculate_iou(box1, box2):
    """This function calculates IOU between two boxes provided we have the following dimensions for each box - (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = box1
    x1_roi, y1_roi, x2_roi, y2_roi = box2

    xi1 = max(x1, x1_roi)
    yi1 = max(y1, y1_roi)
    xi2 = min(x2, x2_roi)
    yi2 = min(y2, y2_roi)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_roi - x1_roi) * (y2_roi - y1_roi)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Load YOLO model
model = YOLO("yolov8m.pt") # Nano is pretty inaccurate most of the time. Never really tried small, but maybe that's good enough. Medium works well.


video_path = "/content/DT-Test.mp4" # This is a path that works, provided you upload the video to Colab's temporary session storage
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get frame dimensions and video info
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    exit()
frame_resized = rescaleFrame(first_frame, scale=0.6)
frame_height, frame_width = frame_resized.shape[:2]
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames at {fps} FPS")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Define window ROI
window_roi = [int(0.5 * frame_width), int(0.1 * frame_height),
              int(frame_width), int(0.5 * frame_height)]  # [x1, y1, x2, y2]

# Set up video writer
output_path = "/content/DT-New-Angle.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Tracking variables
car_dwell_times = {}  # ID -> (centroid, frames)
counted_cars = set()
next_id = 0
dwell_threshold = 5 * fps  # 5 seconds in frames
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = rescaleFrame(frame, scale=0.6)

    # Run YOLO detection
    results = model(frame_resized, verbose=False)

    # Current frame’s car detections
    current_cars = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls) == 2 and box.conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Match ID
                matched_id = None
                for old_id, (old_centroid, _) in car_dwell_times.items():
                    dist = ((centroid[0] - old_centroid[0])**2 + (centroid[1] - old_centroid[1])**2)**0.5
                    if dist < 50:
                        matched_id = old_id
                        break

                if matched_id is None:
                    matched_id = next_id
                    next_id += 1

                current_cars[matched_id] = (centroid, [x1, y1, x2, y2])

    # Update dwell times
    for car_id in list(car_dwell_times.keys()):
        if car_id not in current_cars:
            del car_dwell_times[car_id]

    for car_id, (centroid, box) in current_cars.items():
        iou = calculate_iou(box, window_roi)
        if iou > 0.3:  # This is ideal for the current ROI.
            car_dwell_times[car_id] = (centroid, car_dwell_times.get(car_id, (centroid, 0))[1] + 1)
        else:
            car_dwell_times[car_id] = (centroid, 0)

        dwell_frames = car_dwell_times[car_id][1]
        if dwell_frames >= dwell_threshold and car_id not in counted_cars:
            counted_cars.add(car_id)
            print(f"Car {car_id} stayed at window for 5+ seconds. Total count: {len(counted_cars)}")

    # Visualization
    cv2.rectangle(frame_resized, (window_roi[0], window_roi[1]),
                  (window_roi[2], window_roi[3]), (0, 255, 255), 2)
    for car_id, (centroid, box) in current_cars.items():
        x1, y1, x2, y2 = box
        iou = calculate_iou(box, window_roi)
        color = (0, 255, 0) if iou > 0.3 else (0, 0, 255)
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_resized, f"ID {car_id} IOU: {iou:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame_resized, f"Count: {len(counted_cars)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write frame to output video
    out.write(frame_resized)

# The final result is along with some other metrics
end_time = time.time()
processing_time = end_time - start_time
print(f"Processed {total_frames} frames in {processing_time:.2f} seconds")
print(f"Average FPS: {total_frames / processing_time:.2f}")
print(f"Video duration: {total_frames / fps:.2f} seconds")
print(f"Total cars exited: {len(counted_cars)}")
cap.release()
out.release()
