import torch
import torch.nn as nn

class LandmarkClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)