import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_dynamic import LSTMClassifier
import json

class SequenceDataset(Dataset):
    def __init__(self, file):
        with open(file) as f:
            self.data = json.load(f)
        classes = sorted({item["class"] for item in self.data})
        self.class_to_id = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx]["data"], dtype=torch.float32)
        label = self.class_to_id[self.data[idx]["class"]]
        return seq, label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

ds = SequenceDataset("data_secvente.json")
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

INPUT_SIZE = len(ds[0][0][0]) 
HIDDEN_SIZE = 128
NUM_CLASSES = len(ds.class_to_id)

model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    for seqs, labels, lengths in loader:
        optimizer.zero_grad()
        out = model(seqs, lengths)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} done")

torch.save(model.state_dict(), "model_dynamic.pth")
