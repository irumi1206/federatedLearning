from torch.utils.data import Dataset

class FEMNISTClientDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["image"], sample["character"]