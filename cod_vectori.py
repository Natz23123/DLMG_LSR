import cv2 as cv
import time
import mediapipe as mp
import numpy as np
import json
from model_vectori import LandmarkClassifier
import torch

with open("data_all.json") as f:
    data = json.load(f)

letters = sorted(list({d["class"] for d in data}))
class_to_id = {c: i for i, c in enumerate(letters)}
id_to_class = {v: k for k, v in class_to_id.items()}

model = LandmarkClassifier(len(letters))
model.load_state_dict(torch.load("model_vectori.pth", map_location="cpu"))
model.eval()

def angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def extract_landmarks(hand_landmarks):
    lm = hand_landmarks.landmark

    WRIST = lm[0]
    THUMB_CMC = lm[1]
    THUMB_MCP = lm[2]
    THUMB_IP = lm[3]
    THUMB_TIP = lm[4]
    INDEX_MCP = lm[5]
    INDEX_PIP = lm[6]
    INDEX_DIP = lm[7]
    INDEX_TIP = lm[8]
    MIDDLE_MCP = lm[9]
    MIDDLE_PIP = lm[10]
    MIDDLE_DIP = lm[11]
    MIDDLE_TIP = lm[12]
    RING_MCP = lm[13]
    RING_PIP = lm[14]
    RING_DIP = lm[15]
    RING_TIP = lm[16]
    PINKY_MCP = lm[17]
    PINKY_PIP = lm[18]
    PINKY_DIP = lm[19]
    PINKY_TIP = lm[20]

    landmarks = [p for p in lm]

    v1 = [
        THUMB_CMC.x - WRIST.x,
        THUMB_CMC.y - WRIST.y,
        THUMB_CMC.z - WRIST.z
    ]

    v2 = [
        THUMB_MCP.x - THUMB_CMC.x,
        THUMB_MCP.y - THUMB_CMC.y,
        THUMB_MCP.z - THUMB_CMC.z
    ]

    v3 = [
        THUMB_IP.x - THUMB_MCP.x,
        THUMB_IP.y - THUMB_MCP.y,
        THUMB_IP.z - THUMB_MCP.z
    ]
    v4 = [
        THUMB_TIP.x - THUMB_IP.x,
        THUMB_TIP.y - THUMB_IP.y,
        THUMB_TIP.z - THUMB_IP.z
    ]
    v5 = [
        INDEX_MCP.x - WRIST.x,
        INDEX_MCP.y - WRIST.y,
        INDEX_MCP.z - WRIST.z
    ]
    v6 = [
        INDEX_PIP.x - INDEX_MCP.x,
        INDEX_PIP.y - INDEX_MCP.y,
        INDEX_PIP.z - INDEX_MCP.z
    ]
    v7 = [
        INDEX_DIP.x - INDEX_PIP.x,
        INDEX_DIP.y - INDEX_PIP.y,
        INDEX_DIP.z - INDEX_PIP.z
    ]
    v8 = [
        INDEX_TIP.x - INDEX_DIP.x,
        INDEX_TIP.y - INDEX_DIP.y,
        INDEX_TIP.z - INDEX_DIP.z
    
    ]

    v9 = [
        MIDDLE_MCP.x - WRIST.x,
        MIDDLE_MCP.y - WRIST.y,
        MIDDLE_MCP.z - WRIST.z
    ]
    v10 = [
        MIDDLE_PIP.x - MIDDLE_MCP.x,
        MIDDLE_PIP.y - MIDDLE_MCP.y,
        MIDDLE_PIP.z - MIDDLE_MCP.z
    ]
    v11 = [
        MIDDLE_DIP.x - MIDDLE_PIP.x,
        MIDDLE_DIP.y - MIDDLE_PIP.y,
        MIDDLE_DIP.z - MIDDLE_PIP.z
    ]
    v12 = [
        MIDDLE_TIP.x - MIDDLE_DIP.x,
        MIDDLE_TIP.y - MIDDLE_DIP.y,
        MIDDLE_TIP.z - MIDDLE_DIP.z
    ]

    v13 = [
        RING_MCP.x - WRIST.x,
        RING_MCP.y - WRIST.y,
        RING_MCP.z - WRIST.z
    ]
    v14 = [
        RING_PIP.x - RING_MCP.x,
        RING_PIP.y - RING_MCP.y,
        RING_PIP.z - RING_MCP.z
    ]
    v15 = [
        RING_DIP.x - RING_PIP.x,
        RING_DIP.y - RING_PIP.y,
        RING_DIP.z - RING_PIP.z
    ]
    v16 = [
        RING_TIP.x - RING_DIP.x,
        RING_TIP.y - RING_DIP.y,
        RING_TIP.z - RING_DIP.z
    ]
    v17 = [
        PINKY_MCP.x - WRIST.x,
        PINKY_MCP.y - WRIST.y,
        PINKY_MCP.z - WRIST.z
    ]
    v18 = [
        PINKY_PIP.x - PINKY_MCP.x,
        PINKY_PIP.y - PINKY_MCP.y,
        PINKY_PIP.z - PINKY_MCP.z
    ]
    v19 = [
        PINKY_DIP.x - PINKY_PIP.x,
        PINKY_DIP.y - PINKY_PIP.y,
        PINKY_DIP.z - PINKY_PIP.z
    ]
    v20 = [
        PINKY_TIP.x - PINKY_DIP.x,
        PINKY_TIP.y - PINKY_DIP.y,
        PINKY_TIP.z - PINKY_DIP.z
    ]

    vect = [
        v1, v2, v3, v4, v5, v6, v7, v8,
        v9, v10, v11, v12, v13, v14, v15, v16,
        v17, v18, v19, v20
    ]        

    THUMB_HOOK_ANGLE = angle(v3, v4)
    THUMB_INDEX_SPREAD_ANGLE = angle(v2, v5)
    INDEX_MIDDLE_SPREAD_ANGLE = angle(v6, v10)
    MIDDLE_RING_SPREAD_ANGLE = angle(v10, v14)
    RING_PINKY_SPREAD_ANGLE = angle(v14, v18)

    angles = [
        THUMB_HOOK_ANGLE,
        THUMB_INDEX_SPREAD_ANGLE,
        INDEX_MIDDLE_SPREAD_ANGLE,
        MIDDLE_RING_SPREAD_ANGLE,
        RING_PINKY_SPREAD_ANGLE
    ]

    all_data = []
    for p in landmarks:
        all_data.extend([p.x, p.y, p.z])
    for v in vect:
        all_data.extend(v)
    all_data.extend(angles)

    return all_data


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
