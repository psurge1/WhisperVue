import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image

# Initialize object detection model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Load MiDaS model for depth estimation
depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
depth_model.eval().to(device)

# Transform for depth estimation
transform = Compose([
    Resize((384, 384)),
    ToTensor(),
    Normalize([0.5], [0.5])
])

def estimate_depth(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = depth_model(input_tensor)
    return depth_map.squeeze().cpu().numpy()

def detect_objects(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, size=640)
    detections = results.pred[0]

    objects = []
    if detections is not None and len(detections) > 0:
        depth_map = estimate_depth(frame)
        for *box, conf, cls in detections:
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            scaled_center_x = int(center_x * (depth_map.shape[1] / frame.shape[1]))
            scaled_center_y = int(center_y * (depth_map.shape[0] / frame.shape[0]))
            depth = depth_map[scaled_center_y, scaled_center_x] if (0 <= scaled_center_x < depth_map.shape[1] and 0 <= scaled_center_y < depth_map.shape[0]) else None
            objects.append({
                "name": label,
                "distance": depth  # You may want to replace depth with a distance calculation
            })

    return objects
