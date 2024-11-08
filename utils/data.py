import numpy as np
import torch
from torch.utils.data import Dataset


def load_data(dataset, val_ratio):
    
    if 'SMD' in dataset:
        d, n1, n2 = dataset.split('_')
        train = np.load(f'./data/{d}/machine-{n1}-{n2}_train.npy').astype(np.float64)
        test = np.load(f'./data/{d}/machine-{n1}-{n2}_test.npy').astype(np.float64)
        labels = np.load(f'./data/{d}/machine-{n1}-{n2}_labels.npy').astype(np.float64)

    else:
        train = np.load(f'./data/{dataset}/train.npy').astype(np.float64)
        test = np.load(f'./data/{dataset}/test.npy').astype(np.float64)
        labels = np.load(f'./data/{dataset}/labels.npy').astype(np.float64)

    # Train-validation split
    val_length = int(train.shape[0] * val_ratio)
    train = train[:-val_length, :]
    val = train[-val_length:, :]

    return train, val, test, labels

    
class TimeDataset(Dataset):
    def __init__(self, args, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.Tensor(labels)
        self.window = args.seq_len
        self.x, self.label = self.process(self.data, self.labels)
    
    def __len__(self):
        return len(self.x)

    def process(self, data, label): ## Windowing process
        x_arr, label_arr = [], []
        T, N = data.shape 

        for t in range(self.window-1, T):
            x = data[t-self.window+1:t+1, :]
            l = label[t, :]
            x_arr.append(x)
            label_arr.append(l)

        x_tensor = torch.stack(x_arr).contiguous()
        label_tensor = torch.stack(label_arr).contiguous()
        return x_tensor, label_tensor

    def __getitem__(self, idx):
        x = self.x[idx]
        label = self.label[idx]
        return x, label

