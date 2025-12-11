import cv2 as cv
import time
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

DATA_DIR = "photo_data"

while True:
    ret, img = cap.read()
    if not ret:
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    cv.imshow("DLMG_LSR", img)

    key = cv.waitKey(1)

    if key == ord("q"):
        break
    if key == ord(" "):
        letter = input("Litera pentru această imagine: ").strip().upper()
        path = os.path.join(DATA_DIR, letter)
        os.makedirs(path, exist_ok=True)

        filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
        cv.imwrite(os.path.join(path, filename), img)
        print("Saved.")

cap.release()
cv.destroyAllWindows()
