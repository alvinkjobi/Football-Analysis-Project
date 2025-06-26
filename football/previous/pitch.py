from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load your trained model
model = YOLO(r"D:\PROJECT\models\best.pt")

# ---------- INPUT PATH ----------
input_path = r"D:\PROJECT\input-videos\match2.mp4"  # Now set to match.mp4

# Check if input is image or video
if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
    results = model(input_path, save=True)
else:
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = os.path.join(
        r"D:\PROJECT\output",
        f"detected_{os.path.basename(input_path)}"
    )
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        int(fps),
        (width, height)
    )

    # Get class names from model
    class_names = model.model.names

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        annotated_frame = frame.copy()

        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int) if hasattr(results.boxes, 'cls') else [0]*len(boxes)

        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = y2  # bottom of the box

            # Get label from class id
            role = class_names[class_id].capitalize() if class_id < len(class_names) else "Player"

            # Ellipse parameters for TV/ground view
            axes_length = ((x2 - x1), max(10, (y2 - y1) // 10))
            angle = 25
            startAngle = -45
            endAngle = 245

            overlay = annotated_frame.copy()
            fill_color = (255, 120, 30)  # FIFA-like color (BGR)
            border_color = (255, 255, 255)  # White border

            cv2.ellipse(
                overlay,
                (center_x, center_y),
                axes_length,
                angle,
                startAngle,
                endAngle,
                fill_color,
                -1
            )
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            cv2.ellipse(
                annotated_frame,
                (center_x, center_y),
                axes_length,
                angle,
                startAngle,
                endAngle,
                border_color,
                3
            )
            # Write role above the ellipse
            cv2.putText(
                annotated_frame,
                role,
                (center_x - 40, center_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                border_color,
                2,
                cv2.LINE_AA
            )

        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")

