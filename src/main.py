from flask import Flask, Response
from flask_cors import CORS
import cv2
import torch
import pyttsx3
import threading
import numpy as np
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image

# Initialize models, devices, and TTS engine
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True).eval().to(device)
engine = pyttsx3.init()

app = Flask(__name__)
CORS(app)

# Global variables
frame = None
processed_frame = None
frame_lock = threading.Lock()
processing_flag = threading.Event()  # Use Event to signal when processing is complete
running = True
message_queue = []

# Define transforms
transform = Compose([Resize((384, 384)), ToTensor(), Normalize([0.5], [0.5])])

def capture_frames():
    global frame
    cap = cv2.VideoCapture(0)
    while running:
        with frame_lock:
            if not processing_flag.is_set():  # Capture a new frame only when previous frame is processed
                ret, new_frame = cap.read()
                if ret:
                    frame = new_frame  # Update the global frame variable
                processing_flag.set()  # Signal that a frame is ready for processing
        cv2.waitKey(10)  # Small delay to control loop speed
    cap.release()

# def speak():
#     while running:
#         if message_queue:
#             message = message_queue.pop(0)
#             engine.say(message)
#             engine.runAndWait()

def estimate_depth(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = depth_model(input_tensor)
    return depth_map.squeeze().cpu().numpy()

def detect_objects():
    global frame, processed_frame
    while running:
        processing_flag.wait()  # Wait until a new frame is captured and ready for processing

        with frame_lock:
            if frame is None:
                continue

            # Convert frame to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect objects
            results = model(frame_rgb, size=640)
            detections = results.pred[0]

            # Initialize message for detected objects
            if detections is not None and len(detections) > 0:
                labels = detections[:, -1]
                objects_detected = ', '.join([model.names[int(label.item())] for label in labels])
                message_queue.append(f"I see {objects_detected}")

                # Estimate depth map
                depth_map = estimate_depth(frame)
                
                # Process results and annotate frame
                for *box, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    depth = depth_map[center_y, center_x] if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1] else None
                    if depth is not None:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"{model.names[int(cls)]} Depth: {depth:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Update processed frame after processing is complete
                processed_frame = frame.copy()

        processing_flag.clear()  # Signal that processing for the frame is complete

def gen():
    global processed_frame
    while True:
        with frame_lock:
            if processed_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                processed_frame = None  # Reset processed_frame to signal a new frame can be captured

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/llm')
def llm():
    return "Hello World"


if __name__ == '__main__':
    # Start threads before running the app
    capture_thread = threading.Thread(target=capture_frames)
    # speech_thread = threading.Thread(target=speak)
    detection_thread = threading.Thread(target=detect_objects)
    capture_thread.start()
    # speech_thread.start()
    detection_thread.start()

    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    finally:
        running = False
        capture_thread.join()
        # speech_thread.join()
        detection_thread.join()
        cv2.destroyAllWindows()
        engine.stop()
