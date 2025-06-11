from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class HuggingFaceFEMNIST(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.to_tensor = ToTensor()
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = self.to_tensor(item["image"])  # converts PIL to tensor
        return img, item["character"]
