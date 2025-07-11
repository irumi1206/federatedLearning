import torch.nn as nn
import torch.nn.functional as F

class CNNFEMNIST(nn.Module):
    def __init__(self):
        super(CNNFEMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 62)  # FEMNIST has 62 classes: 10 digits + 26 lowercase + 26 uppercase

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def get_model():
    return CNNFEMNIST()