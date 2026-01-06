import cv2 as cv
import os
import mediapipe as mp
import json
import numpy as np

def angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos, -1.0, 1.0))

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
    
        vect = []
        for p in lm.landmark:
            vect.extend([p.x, p.y, p.z])

        WRIST = lm.landmark[0]
        THUMB_MCP = lm.landmark[2]
        THUMB_TIP = lm.landmark[4]
        INDEX_MCP = lm.landmark[5]

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
        vect.append(angle_thumb_palm)

        THUMB_CMC = lm.landmark[1]

        v1 = [
            THUMB_MCP.x - THUMB_CMC.x,
            THUMB_MCP.y - THUMB_CMC.y,
            THUMB_MCP.z - THUMB_CMC.z
        ]

        v2 = [
            THUMB_TIP.x - THUMB_MCP.x,
            THUMB_TIP.y - THUMB_MCP.y,
            THUMB_TIP.z - THUMB_MCP.z
        ]

        angle_thumb_bend = angle(v1, v2)
        vect.append(angle_thumb_bend)

        dataset.append({"class": letter, "landmarks": vect})

with open(output_file, "w") as f:
    json.dump(dataset, f, indent=4)