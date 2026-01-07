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

def extract_data(lm):
    WRIST = lm.landmark[0]
    THUMB_CMC = lm.landmark[1]
    THUMB_MCP = lm.landmark[2]
    THUMB_IP = lm.landmark[3]
    THUMB_TIP = lm.landmark[4]
    INDEX_MCP = lm.landmark[5]
    INDEX_PIP = lm.landmark[6]
    INDEX_DIP = lm.landmark[7]
    INDEX_TIP = lm.landmark[8]
    MIDDLE_MCP = lm.landmark[9]
    MIDDLE_PIP = lm.landmark[10]
    MIDDLE_DIP = lm.landmark[11]
    MIDDLE_TIP = lm.landmark[12]
    RING_MCP = lm.landmark[13]
    RING_PIP = lm.landmark[14]
    RING_DIP = lm.landmark[15]
    RING_TIP = lm.landmark[16]
    PINKY_MCP = lm.landmark[17]
    PINKY_PIP = lm.landmark[18]
    PINKY_DIP = lm.landmark[19]
    PINKY_TIP = lm.landmark[20]

    landmarks = [p for p in lm.landmark]

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

    scale = np.linalg.norm(
        [
            MIDDLE_MCP.x - WRIST.x,
            MIDDLE_MCP.y - WRIST.y,
            MIDDLE_MCP.z - WRIST.z,
        ]
    )

    if scale < 1e-6:
        scale = 1.0

    all_data = []
    for p in landmarks:
        all_data.extend(
            [
                (p.x - WRIST.x) / scale,
                (p.y - WRIST.y) / scale,
                (p.z - WRIST.z) / scale,
            ]
        )
    for v in vect:
        all_data.extend(v)
    all_data.extend(angles)
    return all_data

def photo_to_vectors():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    DATA_DIR = "photo_data"
    output_file = "data_all.json"

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

            all_data = extract_data(lm)

            dataset.append({"class": letter, "data": all_data})

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)