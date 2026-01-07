import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from model_dynamic import LSTMClassifier  # modelul LSTM pentru dinamice

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

# --- CONFIGURARE ---
SEQ_WINDOW = 15      # câte frame-uri în buffer pentru secvență
THUMB_PINKY_DIST = 0.2  # prag pentru "mâna deschisă"
DEVICE = "cpu"       # sau "cuda" dacă ai GPU

# --- INIȚIALIZARE MODEL ---
INPUT_SIZE = len(extract_data(some_hand_object))  # exact cum e la training
HIDDEN_SIZE = 128  # cum ai folosit la training
NUM_CLASSES = 2     # I și J

dynamic_model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
dynamic_model.load_state_dict(torch.load("model_dynamic.pth", map_location="cpu"))
dynamic_model.eval()

id_to_class = {0: "I", 1: "J"}

# --- INIȚIALIZARE CAMERA ---
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

# --- BUFFER PENTRU SECVENȚE ---
seq_buffer = deque(maxlen=SEQ_WINDOW)

# --- FUNCȚIE EXTRACȚIE VECTOR ---
def extract_landmarks(hand_landmarks):
    lm = hand_landmarks.landmark
    vect = []
    for p in lm:
        vect.extend([p.x, p.y, p.z])
    return np.array(vect, dtype=np.float32)

# --- LOOP PRINCIPAL ---
while True:
    ret, img = cap.read()
    if not ret:
        break

    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)

    do_prediction = False
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
        hand1, hand2 = results.multi_hand_landmarks[:2]

        # deschidere mâna 2
        lm2 = hand2.landmark
        thumb_tip = np.array([lm2[4].x, lm2[4].y])
        pinky_tip = np.array([lm2[20].x, lm2[20].y])
        dist = np.linalg.norm(thumb_tip - pinky_tip)

        if dist > THUMB_PINKY_DIST:
            do_prediction = True

    if do_prediction:
        hand_vect = extract_landmarks(hand1)
        seq_buffer.append(torch.tensor(hand_vect))

        if len(seq_buffer) == SEQ_WINDOW:
            batch = torch.stack(list(seq_buffer)).unsqueeze(0)  # (1, seq_len, input_size)
            lengths = torch.tensor([SEQ_WINDOW])

            with torch.no_grad():
                out = dynamic_model(batch, lengths)
                probs = torch.softmax(out, dim=1)
                pred = int(torch.argmax(probs, dim=1).item())
                letter = id_to_class[pred]

            cv.putText(img, f"DYNAMIC: {letter}", (10, 130),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # desenăm mâinile
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    cv.imshow("Dynamic LSR", img)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
