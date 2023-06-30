# to dataset
from torch.utils.data import Dataset, DataLoader

class TraceDataset(Dataset):
    def __init__(self, traces_x, traces_y):
        self.traces_x = traces_x
        self.traces_y = traces_y
    
    def __len__(self):
        return len(self.traces_x)
    
    def __getitem__(self, idx):
        return self.traces_x[idx], self.traces_y[idx]