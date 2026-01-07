import cv2 as cv
import time
import mediapipe as mp
import numpy as np
import json
from model_mlp import LandmarkClassifier
import torch

with open("data_landmarks.json") as f:
    data = json.load(f)

letters = sorted(list({d["class"] for d in data}))
class_to_id = {c: i for i, c in enumerate(letters)}
id_to_class = {v: k for k, v in class_to_id.items()}

model = LandmarkClassifier(len(letters))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

def angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def extract_landmarks(hand_landmarks):
    lm = hand_landmarks.landmark
    vect = []

    for p in lm:
        vect.extend([p.x, p.y, p.z])

    # ---- unghi thumb vs palm ----
    WRIST = lm[0]
    THUMB_MCP = lm[2]
    THUMB_TIP = lm[4]
    INDEX_MCP = lm[5]

    v_thumb = [
        THUMB_TIP.x - THUMB_MCP.x,
        THUMB_TIP.y - THUMB_MCP.y,
        THUMB_TIP.z - THUMB_MCP.z
    ]

    v_palm = [
        INDEX_MCP.x - WRIST.x,
        INDEX_MCP.y - WRIST.y,
        INDEX_MCP.z - WRIST.z
    ]

    angle_thumb_palm = angle(v_thumb, v_palm)

    # ---- unghi cârlig ----
    THUMB_IP = lm[3]

    v1 = [
        THUMB_IP.x - THUMB_MCP.x,
        THUMB_IP.y - THUMB_MCP.y,
        THUMB_IP.z - THUMB_MCP.z
    ]

    v2 = [
        THUMB_TIP.x - THUMB_MCP.x,
        THUMB_TIP.y - THUMB_MCP.y,
        THUMB_TIP.z - THUMB_MCP.z
    ]

    angle_thumb_bend = angle(v1, v2)

    vect.extend([angle_thumb_palm, angle_thumb_bend])

    return vect


cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mpDraw = mp.solutions.drawing_utils

prev = 0
paused = False
frozen_img = None
while True:
    if not paused:
        ret, img = cap.read()
        if not ret:
            break
        frozen_img = img.copy()
    else:
        if frozen_img is None:
            continue
        img = frozen_img.copy()

    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    hands_res = hands.process(rgb)

    top5_lines = None
    if hands_res.multi_hand_landmarks:
        hand = hands_res.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

        vect = extract_landmarks(hand)
        vect = torch.tensor(vect, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            print(vect.shape)
            out = model(vect)
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1).item()

        if paused:
            k = min(5, probs.numel())
            vals, idxs = torch.topk(probs[0], k=k)
            top5_lines = [
                f"{i+1}. {id_to_class[idxs[i].item()]}: {vals[i].item()*100:.1f}%"
                for i in range(k)
            ]

        letter = id_to_class[pred]

        cv.putText(img, f"{letter}", (10, 130),
               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    if paused:
        cv.putText(img, "PAUSED (press P to resume)", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if top5_lines:
            y = 60
            for line in top5_lines:
                cv.putText(img, line, (10, y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y += 26

    now = time.time()
    fps = 1 / (now - prev) if prev else 0
    prev = now

    cv.putText(img, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("DLMG_LSR", img)

    key = cv.waitKey(1) & 0xFF
    if key in (ord("p"), ord("P")):
        paused = not paused
    elif key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
