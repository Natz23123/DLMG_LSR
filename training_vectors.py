import json
import torch
from torch.utils.data import Dataset, DataLoader
from model_vectori import LandmarkClassifier
import torch.nn as nn
import torch.optim as optim

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

    for epoch in range(50):
        for x, y in dl:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        print(epoch, loss.item())

    torch.save(model.state_dict(), "model_vectori.pth")
