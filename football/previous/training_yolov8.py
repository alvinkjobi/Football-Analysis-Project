from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from collections import Counter
import math

# ------------------- CONFIG -------------------
model_path = r"D:\PROJECT\models\best.pt"
input_video_path = r"D:\PROJECT\input-videos\match2.mp4"
output_video_path = r"D:\PROJECT\output\team_speed_tracking_filtered.mp4"

# ------------------- SETUP -------------------
model = YOLO(model_path)
CLASS_NAMES = model.model.names
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    int(fps),
    (width, height)
)

player_memory = {}
next_player_id = 0
MAX_MISSED = 10
MATCH_DISTANCE = 50
team_colors = {"A": (0, 255, 0), "B": (255, 0, 0)}

# ------------------- HELPERS -------------------
def euclidean_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def generate_id():
    global next_player_id
    pid = f"p{next_player_id}"
    next_player_id += 1
    return pid

# ------------------- MAIN LOOP -------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    current_players = []

    # STEP 1: Detect and extract jersey color using negative imaging
    for xyxy, class_id in zip(detections.xyxy, detections.class_id):
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = CLASS_NAMES.get(class_id, "unknown")
        if class_name != "player":
            continue

        crop = frame[int(y1):int(y1 + (y2 - y1) // 2), int(x1):int(x2)]
        if crop.size == 0:
            continue

        inverted_crop = 255 - crop
        mean_color = np.mean(inverted_crop.reshape(-1, 3), axis=0)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        current_players.append((mean_color, (x1, y1, x2, y2), center))

    # STEP 2: Team detection via KMeans clustering
    if not player_memory and len(current_players) >= 6:
        features = np.array([info[0] for info in current_players])
        kmeans = KMeans(n_clusters=3, n_init="auto").fit(features)
        labels = kmeans.labels_

        # Filter referees (minor cluster)
        label_counts = Counter(labels)
        top_two = [label for label, _ in label_counts.most_common(2)]
        label_map = {top_two[0]: "A", top_two[1]: "B"}

        for i, (_, _, center) in enumerate(current_players):
            label = labels[i]
            if label in label_map:
                pid = generate_id()
                player_memory[pid] = {
                    "pos": center,
                    "team": label_map[label],
                    "missed": 0,
                    "speed": 0.0
                }

    else:
        matched_ids = set()
        updated_memory = {}

        for _, bbox, center in current_players:
            best_match = None
            best_dist = MATCH_DISTANCE

            for pid, data in player_memory.items():
                dist = euclidean_dist(center, data["pos"])
                if dist < best_dist:
                    best_dist = dist
                    best_match = pid

            if best_match:
                prev = player_memory[best_match]["pos"]
                speed = euclidean_dist(center, prev) * fps
                updated_memory[best_match] = {
                    "pos": center,
                    "team": player_memory[best_match]["team"],
                    "missed": 0,
                    "speed": round(speed, 1)
                }
                matched_ids.add(best_match)

        for pid in player_memory:
            if pid not in matched_ids:
                if player_memory[pid]["missed"] < MAX_MISSED:
                    updated_memory[pid] = {
                        "pos": player_memory[pid]["pos"],
                        "team": player_memory[pid]["team"],
                        "missed": player_memory[pid]["missed"] + 1,
                        "speed": player_memory[pid]["speed"]
                    }

        player_memory = updated_memory

    # STEP 3: Draw team info on frame
    for pid, data in player_memory.items():
        x, y = data["pos"]
        team = data["team"]
        speed = data["speed"]
        color = team_colors[team]
        cv2.circle(frame, (x, y + 30), 6, color, -1)
        label = f"Team {team} | {speed} px/s"
        cv2.putText(frame, label, (x - 30, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

# ------------------- CLEANUP -------------------
cap.release()
out.release()
