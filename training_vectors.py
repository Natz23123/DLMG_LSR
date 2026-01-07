import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from model_vectori import LandmarkClassifier
import torch.nn as nn
import torch.optim as optim

def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:d}m {s:02d}s"

def train_vectors():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class LandmarksDS(Dataset):
        def __init__(self, json_path):
            with open(json_path) as f:
                self.data = json.load(f)
            
            #mapam A,B,C -> 0,1,2,...
            letters = sorted(list({d["class"] for d in self.data}))
            self.class_to_id = {c: i for i, c in enumerate(letters)}

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            x = torch.tensor(item["data"], dtype=torch.float32)
            y = torch.tensor(self.class_to_id[item["class"]])
            return x, y

    ds = LandmarksDS("data_all.json")
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    num_classes = len(ds.class_to_id)
    model = LandmarkClassifier(num_classes)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    total_epochs = 50
    batches_per_epoch = len(dl)
    total_batches = total_epochs * batches_per_epoch if batches_per_epoch else 0

    started = time.time()
    processed_batches = 0
    last_update = started

    for epoch in range(total_epochs):
        epoch_start = time.time()
        for batch_idx, (x, y) in enumerate(dl, start=1):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            processed_batches += 1
            now = time.time()
            if now - last_update >= 1 or batch_idx == batches_per_epoch:
                elapsed = now - started
                avg_per_batch = (elapsed / processed_batches) if processed_batches else 0
                remaining = (total_batches - processed_batches) * avg_per_batch if total_batches else 0
                pct = (processed_batches / total_batches * 100) if total_batches else 0
                print(
                    f"[training] Epoch {epoch+1}/{total_epochs} | Batch {batch_idx}/{batches_per_epoch} | Loss: {loss.item():.4f} | {processed_batches}/{total_batches} ({pct:5.1f}%) | Elapsed: {_format_seconds(elapsed)} | ETA: {_format_seconds(remaining)}",
                    flush=True,
                )
                last_update = now

        epoch_time = time.time() - epoch_start
        print(f"[training] Completed epoch {epoch+1}/{total_epochs} in {_format_seconds(epoch_time)} | Last loss: {loss.item():.4f}", flush=True)

    torch.save(model.state_dict(), "model_vectori.pth")
    total_time = time.time() - started
    print(f"[training] Done. Saved model to 'model_vectori.pth'. Total time: {_format_seconds(total_time)}", flush=True)
