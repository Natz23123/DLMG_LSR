# Quick Wins (Presentation-Ready Improvements)

This file lists fast, high-impact improvements you can make before the presentation. It focuses on the current workflow scripts (`train_vectors` and `cod_vectori`) and keeps scope small (1–3 hours each).

## 1) Add a clear "How to Run" section (README)
**Impact:** Big (demo goes smoothly). **Effort:** Low.
- Add a minimal runbook:
  - environment setup (`python -m venv`, `pip install -r requirements.txt`)
  - where data lives (e.g., `photo_data/`)
  - command(s) to generate vectors and train
  - expected outputs (saved model, metrics, logs)
- Include a “CPU-only quick demo” mode if possible.

## 2) Provide a single entrypoint script
**Impact:** Big. **Effort:** Low–Medium.
Create `run_pipeline.py` (or `main.py`) that:
1. Generates/loads landmarks
2. Builds vectors
3. Trains the model
4. Saves artifacts

Even if the underlying code remains the same, a single entrypoint makes the project feel complete.

## 3) Make results reproducible (seed + deterministic flags)
**Impact:** Medium–High. **Effort:** Low.
- Set seeds in one place (`random`, `numpy`, `torch`).
- Log the seed used.
- If using PyTorch, consider deterministic options for the demo.

## 4) Add lightweight metrics + plots
**Impact:** High (presentation visuals). **Effort:** Low–Medium.
- Print train/val loss and at least one metric (accuracy / MAE / R^2—whatever matches the task).
- Save a simple plot (loss curve) to `outputs/`.
- If classification: confusion matrix.

## 5) Add basic logging (replace scattered prints)
**Impact:** Medium. **Effort:** Low.
- Use Python `logging` with levels (INFO, WARNING).
- Log key pipeline stages: dataset size, vector dims, epochs, final metric.

## 6) Validate inputs early (friendlier errors)
**Impact:** Medium. **Effort:** Low.
Add checks like:
- Data folder exists and is non-empty
- Vector dimensions match model input
- `model.pth` loading errors provide actionable guidance

## 7) Put outputs in a consistent folder
**Impact:** Medium. **Effort:** Low.
- Create `outputs/` for:
  - trained models
  - metrics JSON
  - plots
- Name artifacts with timestamps (or run IDs).

## 8) Add a short demo notebook (optional)
**Impact:** High for live demo. **Effort:** Medium.
- `demo.ipynb` showing:
  - load one image
  - extract landmarks
  - vectorize
  - run inference
  - show output

If time is tight, keep it to 5–10 cells.

## 9) Explain the pipeline in one diagram
**Impact:** High (clarity). **Effort:** Low.
Add a simple diagram to the repo (Mermaid in Markdown is fastest):
- Photo → landmarks → vector → MLP → output

## 10) Clean up deprecated files (presentation hygiene)
**Impact:** Medium. **Effort:** Low.
- Add a `deprecated/` folder and move old scripts there, or add a banner comment at top:
  - “DEPRECATED: kept for reference; use train_vectors.py + cod_vectori.py instead.”

## 11) Document shapes and assumptions
**Impact:** Medium–High. **Effort:** Low.
Add a short section (README or docstring):
- landmark format
- vector length
- model input/output shapes
- what the labels represent

## 12) Add a small test / smoke check
**Impact:** Medium. **Effort:** Low.
Create a tiny script `smoke_test.py` that:
- loads 1–2 images
- runs vectorization
- runs one forward pass
- asserts output shape

This reduces demo risk.

---

## Suggested "Presentation Narrative" (30–60 seconds)
1. Problem: turn images into structured face/landmark vectors.
2. Method: extract landmarks → vectorize → train an MLP.
3. Result: show metric + plot, and a quick inference example.
4. Next steps: better model, more data, hyperparameter tuning, and improved evaluation.
