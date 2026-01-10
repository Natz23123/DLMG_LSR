# DLMG_LSR — Hand Sign Letter Recognition

This project recognizes alphabet letters from hand signs using MediaPipe hand landmarks and a lightweight PyTorch classifier. It includes data extraction from images, model training, and real‑time inference from a webcam.

## Overview
- Purpose: Classify hand signs for letters (including some requiring movement) into alphabet classes.
- Approach: Extract a compact, pose‑invariant feature vector from MediaPipe's 21 hand landmarks, then train a small MLP classifier.
- Main stages:
  1. Dataset vectorization from images in [photo_data/](photo_data)
  2. Model training producing [model_vectori.pth](model_vectori.pth)
  3. Real‑time inference via webcam

## Repository Layout
- [photo_to_vectors.py](photo_to_vectors.py): Converts images into feature vectors and saves [data_all.json](data_all.json).
- [training_vectors.py](training_vectors.py): Trains the classifier on the vectors and saves [model_vectori.pth](model_vectori.pth).
- [cod_vectori.py](cod_vectori.py): Runs webcam inference and overlays predictions.
- [model_vectori.py](model_vectori.py): Defines the `LandmarkClassifier` neural network.
- [photo_data/](photo_data): Image dataset organized by class folders (e.g., `A`, `B`, `C`, and movement variants like `J (are movement)`).
- [requirements.txt](requirements.txt): Python dependencies.
- [deprecated/](deprecated): Earlier/alternative implementations (MLP variants, data builders, etc.).

## Data & Feature Engineering
MediaPipe Hands yields 21 landmarks per hand: WRIST + 4 joints × 5 fingers = 21. Each landmark provides 3D coordinates `(x, y, z)` in normalized image space.

This project builds a 128‑dimensional feature vector per image:
- 63 values: normalized landmark coordinates relative to the wrist
  - For each landmark `p`, compute `p' = (p − WRIST) / scale`
  - `scale = ||MIDDLE_MCP − WRIST||` for coarse size normalization and pose invariance
- 60 values: 20 bone vectors (3D) along finger chains
  - Thumb chain: `WRIST→THUMB_CMC→THUMB_MCP→THUMB_IP→THUMB_TIP` (4 vectors)
  - Index, Middle, Ring, Pinky chains similarly from MCP→PIP→DIP→TIP (4 vectors each)
  - Plus MCP vectors from WRIST to each finger MCP (5 vectors)
- 5 values: inter‑segment angles capturing finger spread and thumb hook
  - `THUMB_HOOK_ANGLE = angle(THUMB_IP→THUMB_MCP, THUMB_TIP→THUMB_IP)`
  - `THUMB_INDEX_SPREAD_ANGLE = angle(THUMB_MCP→THUMB_CMC, INDEX_MCP→WRIST)`
  - `INDEX_MIDDLE_SPREAD_ANGLE = angle(INDEX_PIP→INDEX_MCP, MIDDLE_PIP→MIDDLE_MCP)`
  - `MIDDLE_RING_SPREAD_ANGLE = angle(MIDDLE_PIP→MIDDLE_MCP, RING_PIP→RING_MCP)`
  - `RING_PINKY_SPREAD_ANGLE = angle(RING_PIP→RING_MCP, PINKY_PIP→PINKY_MCP)`

Angle function (with numerical safety):
- `angle(v1, v2) = arccos( clamp( (v1·v2) / (||v1||·||v2||), -1, 1 ) )`

These features are computed identically in both [photo_to_vectors.py](photo_to_vectors.py) and [cod_vectori.py](cod_vectori.py), ensuring consistency between training and inference.

### Dataset Generation (images → vectors)
Implemented in [photo_to_vectors.py](photo_to_vectors.py):
- Scans all class folders under [photo_data/](photo_data).
- Loads each image with OpenCV and runs MediaPipe Hands in `static_image_mode=True`, `max_num_hands=1`.
- If a hand is detected, extracts the 128‑D feature vector and appends `{ "class": <folder>, "data": <vector> }` to the dataset.
- Skips unreadable images or those without detectable hands.
- Saves all vectors to [data_all.json](data_all.json) with helpful progress output.

Resulting JSON structure:
```json
[
  { "class": "A", "data": [ /* 128 floats */ ] },
  { "class": "B", "data": [ /* 128 floats */ ] },
  ...
]
```

## Model & Training
Defined in [model_vectori.py](model_vectori.py):
- `LandmarkClassifier(num_classes)` — a simple MLP:
  - `Linear(128→256) + ReLU`
  - `Linear(256→128) + ReLU`
  - `Linear(128→num_classes)`

Training loop in [training_vectors.py](training_vectors.py):
- Loads [data_all.json](data_all.json) into a `Dataset`, maps class labels to IDs (sorted alphabetically), and batches with `DataLoader(batch_size=8, shuffle=True)`.
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam(lr=1e-3)`
- Epochs: `50`
- Device selection: detects CUDA, but current code trains on CPU tensors; adjust if needed.
- Saves weights to [model_vectori.pth](model_vectori.pth) and prints ETA progress.

Notes:
- No explicit validation split is implemented; consider adding train/val split for monitoring and early stopping.
- No data augmentation; for robustness, consider mirroring, slight rotations, or synthetic landmark jitter.
- Consider moving model and batches to GPU for speed if available.

## Inference (Webcam)
Implemented in [cod_vectori.py](cod_vectori.py):
- Loads [model_vectori.pth](model_vectori.pth) and class mapping from [data_all.json](data_all.json).
- Opens the default camera (640×360), detects up to 2 hands (MediaPipe stream mode), and draws landmark skeleton.
- Computes the same 128‑D feature vector as in training and runs the classifier.
- Overlays the top prediction; when paused, also shows top‑5 with probabilities.

Controls:
- `P`: Pause/resume stream; in pause mode, top‑5 predictions are displayed.
- `Q`: Quit the application.

## Technology Stack
- Core: Python 3.11, PyTorch, NumPy
- Vision & Landmarks: OpenCV, MediaPipe Hands
- Visualization: OpenCV (overlay, FPS)
- Utilities: JSON, standard library
- See [requirements.txt](requirements.txt) for exact versions (torch, torchvision, mediapipe, opencv, numpy, scipy, matplotlib, etc.).

## Setup & Usage
Below are recommended steps on Windows PowerShell.

### 1) Create environment and install dependencies
```powershell
# Optional: create and activate a virtual environment
python -m venv .venv
. .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2) Build the vector dataset from images
```powershell
python photo_to_vectors.py
```
- Output: [data_all.json](data_all.json)
- If some images don't contain detectable hands, they're skipped.

### 3) Train the classifier
```powershell
python training_vectors.py
```
- Output: [model_vectori.pth](model_vectori.pth)
- Console shows running loss, elapsed time, and ETA.

### 4) Run real‑time inference
```powershell
python cod_vectori.py
```
- A window opens showing the camera feed, landmarks, FPS, and the predicted letter.
- Press `P` to pause and see top‑5 probabilities; press `Q` to quit.

## Data Organization
- Place images under [photo_data/](photo_data) in subfolders named by class labels.
- Movement‑required letters (e.g., `J`, `Q`, `Y`, `Z`) have dedicated folders that reflect motion context; current pipeline uses static frames.
- Recommendations:
  - Ensure consistent lighting and hand orientation.
  - Use high‑contrast backgrounds to help detector stability.

## Design Rationale
- Landmark normalization relative to the wrist and scaled by `||MIDDLE_MCP − WRIST||` improves invariance to hand size and distance to camera.
- Bone vectors and inter‑finger angles add relational geometry that complements raw landmark positions.
- A small MLP is sufficient due to compact, informative features and offers fast inference on CPU.

## Limitations & Next Steps
- Movement letters are only inferred from single frames; motion features (temporal smoothing or sequence models) would improve accuracy.
- No validation split or metrics; add `train/val/test` partitions and track accuracy/F1.
- Consider: class balancing, calibration (temperature scaling), and confidence thresholds.
- GPU training path: move `model`, `x`, and `y` to `device` in [training_vectors.py](training_vectors.py) for speed.

## Troubleshooting
- Camera not opening: verify device index in OpenCV (`cv.VideoCapture(0)`), and that no other app is using the camera.
- No landmarks detected: improve lighting, show your palm clearly, and ensure the hand is fully in frame.
- Import errors: confirm the virtual environment is active and `pip install -r requirements.txt` succeeded.
- Model not loading: ensure [model_vectori.pth](model_vectori.pth) exists and was trained against your current [data_all.json](data_all.json).

## License & Attribution
- MediaPipe Hands by Google; OpenCV community contributors.
- This repository includes original code for feature extraction, training, and inference of hand sign classification.
