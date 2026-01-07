import torch
import torch.nn as nn

class LandmarkClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 130),
            nn.ReLU(),
            nn.Linear(130, 63),
            nn.ReLU(),
            nn.Linear(63, num_classes)
        )

    def forward(self, x):
        return self.net(x)
