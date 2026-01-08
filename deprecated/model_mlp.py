import torch
import torch.nn as nn

class LandmarkClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 130),
            nn.ReLU(),
            nn.Linear(130, 65),
            nn.ReLU(),
            nn.Linear(65, num_classes)
        )

    def forward(self, x):
        return self.net(x)
