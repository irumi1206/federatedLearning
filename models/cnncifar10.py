import torch.nn as nn
import torch.nn.functional as F

# About 33MB
class CNNCIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # input: [B, 3, 32, 32]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> [B, 64, 32, 32]
        self.pool = nn.MaxPool2d(2, 2)                            # -> [B, 64, 16, 16]
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # [B, 32, 32, 32]
        x = F.relu(self.conv2(x))      # [B, 64, 32, 32]
        x = self.pool(x)               # [B, 64, 16, 16]
        x = self.dropout(x)
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                # logits
        return x

def get_model():
    return CNNCIFAR10()