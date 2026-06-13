from torch.utils.data.dataset import Dataset
import torch
class TargetDataset(Dataset):
    def __init__(self, target_data: dict ):
        self.data = target_data
        self.keys = ["target_values","values","policies","advantages","actions","IS_weight"] 
    
    def __len__(self):
        return self.data["target_values"].size(0)
    
    def __getitem__(self, index):
        data = {}
        for k in self.keys:
            if k in self.data:
                data[k] = self.data[k][index]
        return data

    
    def collate_fn(self,samples):
        targets = {}
        for k in self.keys:
            if k in self.data:
                targets[k] = []
        for sample in samples:
            for k,v in sample.items():
                targets[k].append(v)
        
        ret = []
        for k,v in targets.items():
            ret.append(torch.stack(v))
        return ret