import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.X = self.dataset.cleaned_text
        self.y = self.dataset.cleaned_summary

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def prepare_data_for_training():
    train_data = ReviewDataset('data/train.csv')
    test_data = ReviewDataset('data/test.csv')
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    return train_loader, test_loader

