import cv2 as cv
import os
import mediapipe as mp
import json

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

        dataset.append({"class": letter, "landmarks": vect})

with open(output_file, "w") as f:
    json.dump(dataset, f, indent=4)