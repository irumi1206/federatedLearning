from models.cnncifar10 import get_model
import torch
import numpy as np
import random
import os

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

fix_seed(42)
model = get_model()
print("Checksum:", sum([p.sum().item() for p in model.parameters()]))