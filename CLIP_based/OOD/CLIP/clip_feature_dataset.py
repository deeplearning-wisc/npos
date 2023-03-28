import torch
from torch.utils.data import Dataset


class clip_feature(Dataset):
    def __init__(self, path='/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/dataset/ImageNet-100/'):
        super().__init__()
        self.features = torch.load(path+'feature.pt')
        self.targets = torch.load(path+'target.pt')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]