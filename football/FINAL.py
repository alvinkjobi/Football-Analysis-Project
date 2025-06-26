from ultralytics import YOLO
import cv2
import os
import numpy as np
import base64
from pymongo import MongoClient
import time

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

input_dir = r"D:\PROJECT\input-videos"
os.makedirs(input_dir, exist_ok=True)
input_path = os.path.join(input_dir, f"mongo_input{ext}")

print(f"Saving decoded file to {input_path} ...")
with open(input_path, "wb") as f:
    f.write(base64.b64decode(data_base64))

model = YOLO(r"D:\PROJECT\models\best.pt")

def draw_rounded_box(img, pt1, pt2, color, thickness, r=10):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def process_and_save(input_path, output_path):
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        frame = cv2.imread(input_path)
        results = model(frame)[0]
        annotated_frame = frame.copy()
        for result in results.boxes:
            xyxy = result.xyxy[0].numpy()
            conf = result.conf[0].item()
            cls = result.cls[0].item()
            label = f"{model.model.names[int(cls)]} {conf:.2f}"
            pt1 = tuple(map(int, xyxy[:2]))
            pt2 = tuple(map(int, xyxy[2:]))
            draw_rounded_box(annotated_frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(annotated_frame, label, (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite(output_path, annotated_frame)
    else:
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            int(fps),
            (width, height)
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            annotated_frame = frame.copy()
            for result in results.boxes:
                xyxy = result.xyxy[0].numpy()
                conf = result.conf[0].item()
                cls = result.cls[0].item()
                label = f"{model.model.names[int(cls)]} {conf:.2f}"
                pt1 = tuple(map(int, xyxy[:2]))
                pt2 = tuple(map(int, xyxy[2:]))
                draw_rounded_box(annotated_frame, pt1, pt2, (255, 0, 0), 2)
                cv2.putText(annotated_frame, label, (pt1[0], pt1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            out.write(annotated_frame)
        cap.release()
        out.release()
        time.sleep(0.5)

output_dir = r"D:\PROJECT\output"
os.makedirs(output_dir, exist_ok=True)

# Always save the output as output.mp4 or output.jpg
if ext == ".mp4":
    output_path = os.path.join(output_dir, "output.mp4")
else:
    output_path = os.path.join(output_dir, "output.jpg")

print(f"Processing file: {input_path}")
process_and_save(input_path, output_path)
print(f"Processed file saved to: {output_path}")

print("Clearing output collection...")
output_collection = db["output"]
output_collection.delete_many({})

# Check file size before encoding
file_size = os.path.getsize(output_path)
max_size = 20 * 1024 * 1024  # 16MB MongoDB document limit

if file_size > max_size:
    print(f"Output file is too large for MongoDB ({file_size} bytes > 16MB). Not uploading to DB.")
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
        output_mimetype = mimetype  # fallback

    output_doc = {
        "filename": os.path.basename(output_path),
        "mimetype": output_mimetype,
        "data": output_base64,
        "uploadedAt": doc.get("uploadedAt")
    }
    output_collection.insert_one(output_doc)
    print("Output file saved to MongoDB 'output' collection as base64.")
