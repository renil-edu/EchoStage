import cv2
import json
import time
from collections import defaultdict, Counter
from deepface import DeepFace

cap = cv2.VideoCapture(0)
print("Press 'a' to start scene, 's' to stop and save scene, 'q' to quit")

scene_count = 0
current_scene_data = []
scene_active = False
scene_start_time = None

def aggregate_scene_data(scene_data, start_time):
    per_second_emotions = defaultdict(list)
    for timestamp, emotions, dominant_emotion in scene_data:
        second = int(timestamp - start_time)
        per_second_emotions[second].append(dominant_emotion)

    per_second_summary = []
    for sec in sorted(per_second_emotions):
        dom_em = Counter(per_second_emotions[sec]).most_common(1)[0][0]
        per_second_summary.append({"time": sec, "dominant_emotion": dom_em})

    count = len(scene_data)
    summed_emotions = defaultdict(float)
    if count == 0:
        average_emotions = {}
    else:
        for _, emotions, _ in scene_data:
            for emo, score in emotions.items():
                summed_emotions[emo] += score
        average_emotions = {emo: summed_emotions[emo] / count for emo in summed_emotions}

    return average_emotions, per_second_summary

def save_scene(scene_num, avg_emotions, per_sec_emotions):
    scene_dict = {
        f"scene_{scene_num}": {
            "average_emotions": avg_emotions,
            "per_second_emotions": per_sec_emotions
        }
    }
    filename = f"scene_{scene_num}.json"
    with open(filename, "w") as f:
        json.dump(scene_dict, f, indent=4)
    print(f"[INFO] Saved scene {scene_num} data to {filename}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if scene_active:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotions = result[0]["emotion"]
        dominant_emotion = result[0]["dominant_emotion"]
        current_time = time.time()
        current_scene_data.append((current_time, emotions, dominant_emotion))
        cv2.putText(frame, dominant_emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Press 'a' to start scene", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Emotion Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        if not scene_active:
            scene_active = True
            scene_start_time = time.time()
            current_scene_data = []
            scene_count += 1
            print(f"[INFO] Scene {scene_count} started")
        else:
            print("[WARN] Scene already active, press 's' to stop it first")
    elif key == ord('s'):
        if scene_active:
            scene_active = False
            avg_emotions, per_sec_emotions = aggregate_scene_data(current_scene_data, scene_start_time)
            save_scene(scene_count, avg_emotions, per_sec_emotions)
            current_scene_data = []
        else:
            print("[WARN] No active scene to stop")

cap.release()
cv2.destroyAllWindows()
