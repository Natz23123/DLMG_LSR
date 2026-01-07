import cv2 as cv
import os
import mediapipe as mp
import json
import numpy as np


def extract_landmarks_63(hand_landmarks) -> list[float]:
    lm = hand_landmarks.landmark

    wrist = lm[0]
    middle_mcp = lm[9]
    scale = np.linalg.norm(
        [
            middle_mcp.x - wrist.x,
            middle_mcp.y - wrist.y,
            middle_mcp.z - wrist.z,
        ]
    )
    if scale < 1e-6:
        scale = 1.0

    vect: list[float] = []
    for p in lm:
        vect.extend(
            [
                (p.x - wrist.x) / scale,
                (p.y - wrist.y) / scale,
                (p.z - wrist.z) / scale,
            ]
        )

    return vect

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATA_DIR = "photo_data"
output_file = "data_landmarks.json"

dataset = []

for letter in os.listdir(DATA_DIR):
    letter_path = os.path.join(DATA_DIR, letter)
    if not os.path.isdir(letter_path):
        continue

    for img_name in os.listdir(letter_path):
        img_path = os.path.join(letter_path, img_name)
        img = cv.imread(img_path)
        if img is None:
            continue

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = hands.process(imgRGB)

        if not res.multi_hand_landmarks:
            continue

        lm = res.multi_hand_landmarks[0]

        vect = extract_landmarks_63(lm)
        dataset.append({"class": letter, "landmarks": vect})

with open(output_file, "w") as f:
    json.dump(dataset, f, indent=4)