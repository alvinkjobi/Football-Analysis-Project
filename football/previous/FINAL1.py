from ultralytics import YOLO
import cv2
import os
import numpy as np
import base64
from pymongo import MongoClient
import time
import sys
import traceback
from collections import defaultdict, deque
import math
import json
from datetime import datetime

# Advanced Tracking Classes
class KalmanFilter:
    """Kalman filter for smooth object tracking"""
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)
        
    def predict(self):
        return self.kalman.predict()
    
    def update(self, measurement):
        self.kalman.correct(measurement)

class TrackedObject:
    """Enhanced object tracking with trajectory and statistics"""
    def __init__(self, obj_id, bbox, class_id, confidence, frame_num):
        self.id = obj_id
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox
        self.center = self.get_center(bbox)
        self.kalman = KalmanFilter()
        
        # Initialize Kalman filter
        self.kalman.kalman.statePre = np.array([self.center[0], self.center[1], 0, 0], dtype=np.float32)
        self.kalman.kalman.statePost = np.array([self.center[0], self.center[1], 0, 0], dtype=np.float32)
        
        # Tracking data
        self.trajectory = deque(maxlen=50)  # Store last 50 positions
        self.trajectory.append((self.center, frame_num))
        self.last_seen = frame_num
        self.age = 0
        self.consecutive_invisible_count = 0
        
        # Performance metrics
        self.total_distance = 0
        self.max_speed = 0
        self.avg_speed = 0
        self.speed_history = deque(maxlen=10)
        
        # Zone analysis
        self.zone_time = defaultdict(int)
        self.current_zone = None
        
    def get_center(self, bbox):
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
    
    def update(self, bbox, confidence, frame_num):
        """Update object with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.center = self.get_center(bbox)
        self.last_seen = frame_num
        self.consecutive_invisible_count = 0
        
        # Update Kalman filter
        measurement = np.array([[np.float32(self.center[0])], [np.float32(self.center[1])]])
        self.kalman.update(measurement)
        
        # Update trajectory
        if len(self.trajectory) > 0:
            last_pos, last_frame = self.trajectory[-1]
            distance = math.sqrt((self.center[0] - last_pos[0])**2 + (self.center[1] - last_pos[1])**2)
            frame_diff = frame_num - last_frame
            
            if frame_diff > 0:
                speed = distance / frame_diff
                self.speed_history.append(speed)
                self.total_distance += distance
                self.max_speed = max(self.max_speed, speed)
                self.avg_speed = sum(self.speed_history) / len(self.speed_history)
        
        self.trajectory.append((self.center, frame_num))
        
    def predict_next_position(self):
        """Predict next position using Kalman filter"""
        prediction = self.kalman.predict()
        return (int(prediction[0]), int(prediction[1]))
    
    def mark_invisible(self):
        """Mark object as invisible for this frame"""
        self.consecutive_invisible_count += 1
        self.age += 1
    
    def is_valid(self):
        """Check if object should continue being tracked"""
        return self.consecutive_invisible_count < 10  # Remove after 10 invisible frames

class AdvancedTracker:
    """Advanced multi-object tracking system"""
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 0
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Field zones for analysis
        self.field_zones = {
            'penalty_box_1': (0, 0, 200, 400),
            'midfield': (200, 0, 600, 400),
            'penalty_box_2': (600, 0, 800, 400)
        }
        
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_zone(self, center, frame_width, frame_height):
        """Determine which zone the object is in"""
        x, y = center
        # Normalize coordinates
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        if norm_x < 0.3:
            return 'left_third'
        elif norm_x > 0.7:
            return 'right_third'
        else:
            return 'middle_third'
    
    def update(self, detections, frame_num, frame_width, frame_height):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all objects as invisible
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].mark_invisible()
                if not self.objects[obj_id].is_valid():
                    del self.objects[obj_id]
            return self.objects  # Always return a dict
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                bbox, class_id, confidence = detection
                self.register(bbox, class_id, confidence, frame_num)
            return self.objects  # Always return a dict
        
        # Get current object centers
        object_centers = []
        object_ids = []
        for obj_id, obj in self.objects.items():
            object_centers.append(obj.center)
            object_ids.append(obj_id)
        
        # Get detection centers
        detection_centers = []
        for detection in detections:
            bbox, class_id, confidence = detection
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            detection_centers.append(center)
        
        # Compute distance matrix
        if len(object_centers) > 0 and len(detection_centers) > 0:
            distances = np.zeros((len(object_centers), len(detection_centers)))
            for i, obj_center in enumerate(object_centers):
                for j, det_center in enumerate(detection_centers):
                    distances[i][j] = self.calculate_distance(obj_center, det_center)
            
            # Hungarian algorithm simulation (simplified)
            used_detection_indices = set()
            used_object_indices = set()
            
            # Find minimum distance matches
            for _ in range(min(len(object_centers), len(detection_centers))):
                min_dist = float('inf')
                min_obj_idx = -1
                min_det_idx = -1
                
                for i in range(len(object_centers)):
                    if i in used_object_indices:
                        continue
                    for j in range(len(detection_centers)):
                        if j in used_detection_indices:
                            continue
                        if distances[i][j] < min_dist and distances[i][j] < self.max_distance:
                            min_dist = distances[i][j]
                            min_obj_idx = i
                            min_det_idx = j
                
                if min_obj_idx != -1 and min_det_idx != -1:
                    # Update existing object
                    obj_id = object_ids[min_obj_idx]
                    bbox, class_id, confidence = detections[min_det_idx]
                    self.objects[obj_id].update(bbox, confidence, frame_num)
                    
                    # Update zone information
                    zone = self.get_zone(self.objects[obj_id].center, frame_width, frame_height)
                    if zone != self.objects[obj_id].current_zone:
                        self.objects[obj_id].current_zone = zone
                    self.objects[obj_id].zone_time[zone] += 1
                    
                    used_object_indices.add(min_obj_idx)
                    used_detection_indices.add(min_det_idx)
            
            # Handle unmatched detections (new objects)
            for i, detection in enumerate(detections):
                if i not in used_detection_indices:
                    bbox, class_id, confidence = detection
                    self.register(bbox, class_id, confidence, frame_num)
            
            # Handle unmatched objects (disappeared objects)
            for i, obj_id in enumerate(object_ids):
                if i not in used_object_indices:
                    self.objects[obj_id].mark_invisible()
                    if not self.objects[obj_id].is_valid():
                        del self.objects[obj_id]
        
        return self.objects  # Always return a dict
    
    def register(self, bbox, class_id, confidence, frame_num):
        """Register new object"""
        self.objects[self.next_id] = TrackedObject(
            self.next_id, bbox, class_id, confidence, frame_num
        )
        self.next_id += 1

def draw_fifa_style_annotations(frame, tracked_objects, model_names, frame_num):
    """Draw FIFA-style professional annotations"""
    annotated_frame = frame.copy()
    
    # FIFA-style colors
    colors = {
        'player': (255, 255, 255),      # White for players
        'ball': (0, 255, 255),          # Yellow for ball
        'referee': (255, 0, 255),       # Magenta for referee
        'goalkeeper': (0, 255, 0)       # Green for goalkeeper
    }
    
    # Team colors (can be customized based on detection logic)
    team_colors = [(255, 100, 100), (100, 100, 255)]  # Light red and light blue
    
    for obj_id, obj in tracked_objects.items():
        bbox = obj.bbox
        class_name = model_names.get(int(obj.class_id), f"class_{int(obj.class_id)}")
        
        # Determine color based on class
        if class_name == 'player':
            # Assign team color based on player ID (simple logic)
            team_color = team_colors[obj_id % 2]
        else:
            team_color = colors.get(class_name, (255, 255, 255))
        
        # Draw FIFA-style bounding box (rounded corners)
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        
        # Draw rounded rectangle
        draw_rounded_box(annotated_frame, pt1, pt2, team_color, 2, r=8)
        
        # Draw center dot (small and precise)
        cv2.circle(annotated_frame, obj.center, 2, team_color, -1)
        
        # FIFA-style information display
        if class_name == 'player':
            # Player info box (FIFA style)
            info_box_width = 120
            info_box_height = 40
            info_x = max(5, pt1[0] - 10)
            info_y = max(info_box_height + 5, pt1[1] - 15)
            
            # Semi-transparent background
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, 
                         (info_x - 5, info_y - info_box_height), 
                         (info_x + info_box_width, info_y + 5), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # Player number/ID (large, prominent)
            cv2.putText(annotated_frame, f"#{obj_id}", 
                       (info_x, info_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, team_color, 2)
            
            # Speed indicator (FIFA style)
            speed_kmh = obj.avg_speed * 3.6 * 0.1  # Convert to approximate km/h
            cv2.putText(annotated_frame, f"{speed_kmh:.1f} km/h", 
                       (info_x, info_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        elif class_name == 'ball':
            # Ball tracking info
            cv2.putText(annotated_frame, "BALL", 
                       (pt1[0], pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Ball speed
            speed_kmh = obj.avg_speed * 3.6 * 0.1
            cv2.putText(annotated_frame, f"{speed_kmh:.1f} km/h", 
                       (pt1[0], pt1[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        elif class_name == 'referee':
            # Referee indicator
            cv2.putText(annotated_frame, "REF", 
                       (pt1[0], pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # FIFA-style HUD elements
    hud_overlay = annotated_frame.copy()
    
    # Top-left match info panel
    cv2.rectangle(hud_overlay, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.addWeighted(hud_overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)
    
    # Match time (simulated)
    match_time = f"{frame_num // 30}'"  # Approximate seconds
    cv2.putText(annotated_frame, match_time, (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Players tracked
    cv2.putText(annotated_frame, f"Players: {len([obj for obj in tracked_objects.values() if model_names.get(int(obj.class_id)) == 'player'])}", 
               (20, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Ball possession indicator (if ball is detected)
    ball_objects = [obj for obj in tracked_objects.values() if model_names.get(int(obj.class_id)) == 'ball']
    if ball_objects:
        cv2.putText(annotated_frame, "BALL IN PLAY", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return annotated_frame

def draw_rounded_box(img, pt1, pt2, color, thickness, r=10):
    """Draw rounded rectangle (FIFA style)"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw lines
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    
    # Draw corner arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def generate_analytics_report(tracked_objects, total_frames, fps):
    """Generate comprehensive analytics report"""
    report = {
        "summary": {
            "total_objects_tracked": len(tracked_objects),
            "total_frames": total_frames,
            "duration_seconds": total_frames / fps if fps > 0 else 0
        },
        "objects": {}
    }
    
    for obj_id, obj in tracked_objects.items():
        report["objects"][obj_id] = {
            "class_id": int(obj.class_id),
            "total_distance": obj.total_distance,
            "max_speed": obj.max_speed,
            "avg_speed": obj.avg_speed,
            "trajectory_length": len(obj.trajectory),
            "zone_time": dict(obj.zone_time),
            "last_seen_frame": obj.last_seen
        }
    
    return report

try:
    print("Connecting to MongoDB...")
    client = MongoClient("mongodb://localhost:27017/")
    db = client["analyser"]
    collection = db["input"]

    print("Fetching latest input document...")
    doc = collection.find_one(sort=[("uploadedAt", -1)])
    if not doc:
        print("No input found in the database.")
        raise Exception("No input found in the database.")

    print("Deleting fetched input document...")
    collection.delete_one({'_id': doc['_id']})

    filename = doc.get("filename", "input_file")
    mimetype = doc.get("mimetype", "")
    data_base64 = doc.get("data", "")

    # Set extension to MP4 for video, JPG for images
    if mimetype.startswith("video"):
        ext = ".mp4"
    elif mimetype.startswith("image"):
        ext = ".jpg"
    else:
        raise Exception("Unsupported mimetype: " + mimetype)

    # Use project-relative paths
    project_root = r"D:\FOOTBALL-ANALYSIS-PROJECT\Football-Analysis-Project"
    input_dir = os.path.join(project_root, "input-videos")
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, f"mongo_input{ext}")

    print(f"Saving decoded file to {input_path} ...")
    with open(input_path, "wb") as f:
        f.write(base64.b64decode(data_base64))

    # Model path inside project
    model_path = os.path.join(project_root, "models", "best.pt")
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please ensure the YOLO weights are present.")
        sys.exit(1)
    
    try:
        print(f"Loading YOLO model from {model_path} ...")
        model = YOLO(model_path)
        print("YOLO model loaded successfully.")
    except Exception as model_exc:
        print(f"ERROR: Failed to load YOLO model from {model_path}: {model_exc}")
        traceback.print_exc()
        sys.exit(1)

    # Initialize advanced tracker
    tracker = AdvancedTracker(max_disappeared=15, max_distance=150)

    def process_and_save_advanced(input_path, output_path):
        """Advanced processing with tracking"""
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"Reading image: {input_path}")
            frame = cv2.imread(input_path)
            if frame is None:
                print(f"ERROR: Could not read image file {input_path}")
                raise Exception(f"Could not read image file {input_path}")
            
            print("Running YOLO inference on image...")
            results = model(frame)[0]
            
            # Convert detections for tracker
            detections = []
            for result in results.boxes:
                xyxy = result.xyxy[0].cpu().numpy()
                conf = result.conf[0].item()
                cls = result.cls[0].item()
                detections.append((xyxy, cls, conf))
            
            # Update tracker
            height, width = frame.shape[:2]
            tracked_objects = tracker.update(detections, 0, width, height)
            if tracked_objects is None:
                tracked_objects = {}  # Ensure it's always a dict
            
            # Draw FIFA-style annotations
            annotated_frame = draw_fifa_style_annotations(frame, tracked_objects, model.names, 0)
            
            cv2.imwrite(output_path, annotated_frame)
            print(f"Annotated image saved to {output_path}")
            
        else:
            print(f"Opening video: {input_path}")
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video properties: width={width}, height={height}, fps={fps}")
            
            # Video writer
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                int(fps),
                (width, height)
            )
            
            frame_count = 0
            all_tracked_objects = {}
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO inference
                results = model(frame)[0]
                
                # Convert detections for tracker
                detections = []
                for result in results.boxes:
                    xyxy = result.xyxy[0].cpu().numpy()
                    conf = result.conf[0].item()
                    cls = result.cls[0].item()
                    detections.append((xyxy, cls, conf))
                
                # Update tracker
                tracked_objects = tracker.update(detections, frame_count, width, height)
                
                # Store all tracked objects for analytics
                for obj_id, obj in tracked_objects.items():
                    if obj_id not in all_tracked_objects:
                        all_tracked_objects[obj_id] = obj
                
                # Draw FIFA-style annotations
                annotated_frame = draw_fifa_style_annotations(frame, tracked_objects, model.names, frame_count)
                
                out.write(annotated_frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")
            
            cap.release()
            out.release()
            print(f"Processed {frame_count} frames and saved video to {output_path}")
            
            # Generate analytics report
            analytics_report = generate_analytics_report(all_tracked_objects, frame_count, fps)
            
            # Save analytics report
            analytics_path = os.path.join(project_root, "output", "analytics_report.json")
            with open(analytics_path, "w") as f:
                json.dump(analytics_report, f, indent=2)
            print(f"Analytics report saved to {analytics_path}")

    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Always save the output as output.mp4 or output.jpg
    if ext == ".mp4":
        output_path = os.path.join(output_dir, "output.mp4")
    else:
        output_path = os.path.join(output_dir, "output.jpg")

    print(f"Processing file with advanced tracking: {input_path}")
    process_and_save_advanced(input_path, output_path)
    print(f"Processed file saved to: {output_path}")

    print("Clearing output collection...")
    output_collection = db["output"]
    output_collection.delete_many({})

    # Check file size before encoding
    if not os.path.isfile(output_path):
        print(f"ERROR: Output file {output_path} was not created.")
        raise Exception(f"Output file {output_path} was not created.")
    
    file_size = os.path.getsize(output_path)
    max_size = 20 * 1024 * 1024  # 20MB limit

    if file_size == 0:
        print(f"ERROR: Output file {output_path} is empty.")
        raise Exception(f"Output file {output_path} is empty.")

    if file_size > max_size:
        print(f"Output file is too large for MongoDB ({file_size} bytes > 20MB). Not uploading to DB.")
        print("You can serve the file directly from disk if needed.")
    else:
        print("Encoding output file as base64...")
        with open(output_path, "rb") as f:
            output_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Set MIME type
        if output_path.lower().endswith('.mp4'):
            output_mimetype = "video/mp4"
        elif output_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            output_mimetype = "image/jpeg"
        else:
            output_mimetype = mimetype

        output_doc = {
            "filename": os.path.basename(output_path),
            "mimetype": output_mimetype,
            "data": output_base64,
            "uploadedAt": doc.get("uploadedAt"),
            "tracking_enabled": True,
            "processed_at": datetime.now().isoformat()
        }
        output_collection.insert_one(output_doc)
        print("Output file with advanced tracking saved to MongoDB 'output' collection.")

except Exception as e:
    print("ERROR:", str(e))
    traceback.print_exc()
    sys.exit(1)