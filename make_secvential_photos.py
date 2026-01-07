import cv2 as cv
import os
from datetime import datetime

DATA_DIR = "dynamic_data"

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

recording = False
frame_idx = 0
save_path = None

print("X = start/stop | SPACE = capture frame | Q = quit")

while True:
    ret, img = cap.read()
    if not ret:
        break

    if recording:
        cv.putText(img, "REC", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("DLMG_LSR_DYNAMIC_REC", img)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("x"):
        if not recording:
            letter = input("Litera dinamica: ").strip().upper()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(DATA_DIR, letter, timestamp)
            os.makedirs(save_path, exist_ok=True)

            frame_idx = 0
            recording = True
            print("Recording ON")
        else:
            recording = False
            print("Recording OFF")

    if key == ord(" ") and recording:
        filename = f"frame_{frame_idx:03d}.jpg"
        cv.imwrite(os.path.join(save_path, filename), img)
        frame_idx += 1
        print(f"Saved {filename}")

cap.release()
cv.destroyAllWindows()
