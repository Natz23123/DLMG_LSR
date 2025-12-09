import cv2 as cv
import time
import mediapipe as mp
import numpy as np

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

mpFaceMesh = mp.solutions.face_mesh
face = mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
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
        for h in hands_res.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, h, mpHands.HAND_CONNECTIONS)

    if face_res.multi_face_landmarks:
        lm = face_res.multi_face_landmarks[0].landmark
        mouth_open = abs(lm[13].y - lm[14].y)

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
