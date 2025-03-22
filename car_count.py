# Install latest ultralytics (no version pin, fetches latest stable)
!pip install ultralytics -q

# Import libraries
import cv2
from ultralytics import YOLO
from google.colab import files
import os

# Hardcoded video path
VIDEO_PATH = "/content/DT-AREA-MORNING.mp4"
OUTPUT_PATH = "drive_thru_output_full.mp4"

# Load YOLOv11 model with tracking enabled
model = YOLO("yolo11x.pt")

# Video processing setup
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Failed to open {VIDEO_PATH}. Check file integrity.")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (frame_width, frame_height))

# Define counting line (10% down, right half)
LINE_Y = int(0.1 * frame_height)  # 10% from top
LINE_X1 = int(0.5 * frame_width)  # Start at half width
LINE_X2 = frame_width             # End at full width

# Vehicle class IDs from COCO (car: 2, bus: 5, truck: 7)
VEHICLE_CLASSES = [2, 5, 7]

# Process the entire video
unique_ids = set()
frame_count = 0
prev_positions = {}  # Track previous y2 positions for crossing detection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw counting line on the frame (blue, thickness 2)
    cv2.line(frame, (LINE_X1, LINE_Y), (LINE_X2, LINE_Y), (255, 0, 0), 2)

    # Add current unique vehicle count at top-left (white text)
    count_text = f"Unique Vehicles: {len(unique_ids)}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Track vehicles with YOLOv11, suppress verbose logs
    results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=0.5, iou=0.7, verbose=False)[0]

    # Annotate frame and count vehicles crossing the line
    if results.boxes.id is not None:  # Check if tracking IDs exist
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        ids = results.boxes.id.cpu().numpy().astype(int)
        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            label = f"vehicle {track_id}"
            # Annotate all vehicles (label below box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Check for line crossing (bottom edge y2 crossing LINE_Y downward)
            prev_y2 = prev_positions.get(track_id, y2)
            if prev_y2 < LINE_Y and y2 >= LINE_Y and x1 >= LINE_X1 and x2 <= LINE_X2:  # Crossed downward within line bounds
                unique_ids.add(track_id)
                print(f"Vehicle {track_id} crossed line at frame {frame_count}. Total unique vehicles: {len(unique_ids)}")
            prev_positions[track_id] = y2  # Update previous position

    # Write annotated frame
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Print total vehicle count
print(f"Total unique vehicles fully crossing top-right line in full video: {len(unique_ids)}")

# Download the output video
if os.path.exists(OUTPUT_PATH):
    files.download(OUTPUT_PATH)
else:
    print(f"Output file {OUTPUT_PATH} not created. Check video processing.")
