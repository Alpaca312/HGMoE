


import os
from torch_geometric.data import Data
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for label in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label)
            if os.path.isdir(label_path):
                for sample_name in os.listdir(label_path):
                    sample_path = os.path.join(label_path, sample_name)
                    if os.path.isdir(sample_path):
                        samples.append(sample_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        data = self._load_data(sample_path)
        return data

    def _load_data(self, sample_path):
        edge_attr = torch.load(os.path.join(sample_path, 'edge_attr.pt'))
        edge_index = torch.load(os.path.join(sample_path, 'edge_index.pt'))
        feature = torch.load(os.path.join(sample_path, 'feature.pt'))
        label = os.path.basename(os.path.dirname(sample_path))
        y = torch.tensor([1], dtype=torch.long) if label == 'ASD' else torch.tensor([0], dtype=torch.long)
        return Data(x=feature, edge_index=edge_index, edge_attr=edge_attr, y=y)

