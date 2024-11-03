import cv2
import torch
import threading
import numpy as np
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image

# Initialize object detection model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Load MiDaS model for depth estimation
depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
depth_model.eval().to(device)

# Variables for threading
frame = None
running = True

# Transform for depth estimation
transform = Compose([
    Resize((384, 384)),  # Resize input for MiDaS
    ToTensor(),  # Convert image to Tensor
    Normalize([0.5], [0.5])  # Normalize input (change based on the MiDaS model)
])

def get_detected_objects(detections, depth_map):
    h_scale = depth_map.shape[0] / frame.shape[0]
    w_scale = depth_map.shape[1] / frame.shape[1]
    objects = []

    for *box, conf, cls in detections:
        label = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)
        
        # Calculate the midpoint (center coordinates) of the bounding box
        center_x = int((x1 + x2) // 2)
        center_y = int((y1 + y2) // 2)
        midpoint = (center_x, center_y)
        
        # Scale midpoint for depth map lookup
        scaled_center_x = int(center_x * w_scale)
        scaled_center_y = int(center_y * h_scale)

        if 0 <= scaled_center_x < depth_map.shape[1] and 0 <= scaled_center_y < depth_map.shape[0]:
            depth = depth_map[scaled_center_y, scaled_center_x]
            distance_in_inches, distance_in_feet = estimate_distance(depth)
        else:
            distance_in_inches, distance_in_feet = None, None

        objects.append({
            "label": label,
            "midpoint": midpoint,
            "distance": distance_in_inches
        })
    
    return objects

def get_objects():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, size=640)
    detections = results.pred[0]

    if detections is not None and len(detections) > 0:
        depth_map = estimate_depth(frame)
        detected_objects = get_detected_objects(detections, depth_map)
    else:
        detected_objects = []

    cap.release()
    return detected_objects  # returns a list of detected objects in {label, box, distance_inches, distance_feet} format

def capture_frames():
    global frame
    cap = cv2.VideoCapture(0)
    while running:
        ret, new_frame = cap.read()
        if ret:
            frame = new_frame  # Update the global frame variable
    cap.release()

def estimate_distance(depth_value):
    # Check if the depth value is within the valid range
    if depth_value <= 0:
        return float('inf'), float('inf')  # Infinite distance if depth value is invalid

    # Calculate the distance based on depth
    if depth_value >= 35:
        distance_in_inches = 36  # Minimum distance at maximum depth
    else:
        # As depth decreases from 35 to 1, the distance should increase from 36 to a higher value
        # Set the maximum distance to be 36 inches when the depth is 35, and decrease exponentially.
        # For depth 1, we want the distance to be a value greater than 36
        min_depth = 1  # Minimum depth
        max_depth = 35  # Maximum depth

        # Define the maximum distance we want at the minimum depth
        max_distance = 100  # Set an upper limit for distance (you can adjust this)

        # Calculate the distance inversely proportional to depth
        distance_in_inches = max_distance * (1 - (depth_value - min_depth) / (max_depth - min_depth))

    # Convert to feet
    distance_in_feet = distance_in_inches / 12  # Convert inches to feet

    return distance_in_inches, distance_in_feet

def estimate_depth(image):
    # Convert the NumPy array (OpenCV format) to a PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
    input_tensor = transform(image_pil).unsqueeze(0).to(device)  # Apply the transformation
    with torch.no_grad():
        depth_map = depth_model(input_tensor)
    return depth_map.squeeze().cpu().numpy()

def detect_objects():
    global frame
    while running:
        if frame is not None:
            # Convert frame to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect objects
            results = model(frame_rgb, size=640)  # Adjust 'size' for input resolution

            # Get the results
            detections = results.pred[0]  # Extract predictions from the first batch

            if detections is not None and len(detections) > 0:
                # Process results
                for *box, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Estimate depth map
                    depth_map = estimate_depth(frame)

                    # Scale the center coordinates to match the depth map
                    scaled_center_x = int(center_x * (depth_map.shape[1] / frame.shape[1]))
                    scaled_center_y = int(center_y * (depth_map.shape[0] / frame.shape[0]))

                    # Ensure that scaled coordinates are within bounds
                    if 0 <= scaled_center_x < depth_map.shape[1] and 0 <= scaled_center_y < depth_map.shape[0]:
                        depth = depth_map[scaled_center_y, scaled_center_x]  # Get the depth value at the scaled center coordinates
                    else:
                        depth = None  # Handle out of bounds gracefully

                    # Estimate distance using the depth value
                    if depth is not None:
                        distance_in_inches, distance_in_feet = estimate_distance(depth)
                        # Draw boxes on detected objects and display the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"{model.names[int(cls)]} Depth: {depth:.2f}m, Distance: {distance_in_inches:.2f}in, {distance_in_feet:.2f}ft", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, f"{model.names[int(cls)]} Depth: N/A", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Show the frame in a window
            cv2.imshow('Live Camera Feed', frame)

        # Use a short wait time to avoid high CPU usage
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start threads
capture_thread = threading.Thread(target=capture_frames)
detection_thread = threading.Thread(target=detect_objects)

# Start the threads
capture_thread.start()
detection_thread.start()

try:
    capture_thread.join()
    detection_thread.join()
except KeyboardInterrupt:
    running = False

# Clean up
running = False
cv2.destroyAllWindows()