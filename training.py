import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_cnn import HandSignCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

data = datasets.ImageFolder("data/", transform=transform)
loader = DataLoader(data, batch_size=32, shuffle=True)

model = HandSignCNN(num_classes=len(data.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

torch.save({
    "model_state": model.state_dict(),
    "classes": data.classes
}, "handsign_model.pt")

print("Model saved as handsign_model.pt")
