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
model.load_state_dict(torch.load("model.pth"))
model.eval()

def extract_landmarks(hand_landmarks):
    vect = []
    for p in hand_landmarks.landmark:
        vect.extend([p.x, p.y, p.z])
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
while True:
    ret, img = cap.read()
    if not ret:
        break

    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    hands_res = hands.process(rgb)
    face_res  = face.process(rgb)

    if hands_res.multi_hand_landmarks:
        hand = hands_res.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

        vect = extract_landmarks(hand)
        vect = torch.tensor(vect, dtype=torch.float32)

        with torch.no_grad():
            out = model(vect)
            pred = out.argmax().item()

        letter = id_to_class[pred]

        cv.putText(img, f"{letter}", (10, 130),
               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    now = time.time()
    fps = 1 / (now - prev) if prev else 0
    prev = now

    cv.putText(img, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("DLMG_LSR", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
