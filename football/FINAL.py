from ultralytics import YOLO
import cv2
import os
import numpy as np
import base64
from pymongo import MongoClient
import time
import sys
import traceback

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
            print(f"Reading image: {input_path}")
            frame = cv2.imread(input_path)
            if frame is None:
                print(f"ERROR: Could not read image file {input_path}")
                raise Exception(f"Could not read image file {input_path}")
            print(f"Image shape: {frame.shape}")
            print("Running YOLO inference on image...")
            try:
                results = model(frame)[0]
            except Exception as infer_exc:
                print(f"ERROR: YOLO inference failed: {infer_exc}")
                traceback.print_exc()
                raise
            annotated_frame = frame.copy()
            found = False
            for result in results.boxes:
                found = True
                xyxy = result.xyxy[0].numpy()
                conf = result.conf[0].item()
                cls = result.cls[0].item()
                label = f"{model.model.names[int(cls)]} {conf:.2f}"
                pt1 = tuple(map(int, xyxy[:2]))
                pt2 = tuple(map(int, xyxy[2:]))
                draw_rounded_box(annotated_frame, pt1, pt2, (255, 0, 0), 2)
                cv2.putText(annotated_frame, label, (pt1[0], pt1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if not found:
                print("WARNING: No objects detected in image.")
            cv2.imwrite(output_path, annotated_frame)
            print(f"Annotated image saved to {output_path}")
        else:
            print(f"Opening video: {input_path}")
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video properties: width={width}, height={height}, fps={fps}")
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                int(fps),
                (width, height)
            )
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    results = model(frame)[0]
                except Exception as infer_exc:
                    print(f"ERROR: YOLO inference failed on frame {frame_count}: {infer_exc}")
                    traceback.print_exc()
                    raise
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
                frame_count += 1
            cap.release()
            out.release()
            print(f"Processed {frame_count} frames and saved video to {output_path}")
            time.sleep(0.5)

    output_dir = os.path.join(project_root, "output")
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
    if not os.path.isfile(output_path):
        print(f"ERROR: Output file {output_path} was not created.")
        raise Exception(f"Output file {output_path} was not created.")
    file_size = os.path.getsize(output_path)
    max_size = 20 * 1024 * 1024  # 16MB MongoDB document limit

    if file_size == 0:
        print(f"ERROR: Output file {output_path} is empty.")
        raise Exception(f"Output file {output_path} is empty.")

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

except Exception as e:
    print("ERROR:", str(e))
    traceback.print_exc()
    sys.exit(1)
