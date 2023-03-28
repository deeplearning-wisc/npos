import torch
from torch.utils.data import Dataset


class clip_feature_file_dataset(Dataset):
    def __init__(self, path='/afs/cs.wisc.edu/u/t/a/taoleitian/private/code/dataset/ImageNet-100/'):
        super().__init__()
        file = open(path, 'r')
        path_list = file.read()
        self.path_list = eval(path_list)
        self.target_list = [torch.tensor(int(i.split('/')[-2])).long() for i in self.path_list]
        #for i in self.target_list:
            #print(i)


    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        feature = torch.load(self.path_list[idx]).squeeze()
        return feature, self.target_list[idx]