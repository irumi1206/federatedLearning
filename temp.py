import torch
import torch.nn as nn
import numpy as np
import random
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

transform = transforms.Compose([transforms.ToTensor()])
testset = CIFAR10(root="./data", train=False, transform=transform, download=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(32*32*3, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Initial accuracy: {correct/total:.4f}")
