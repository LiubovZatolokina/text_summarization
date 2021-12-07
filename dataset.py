import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class ReviewDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.X = self.dataset.cleaned_text[:150]
        self.y = self.dataset.cleaned_summary[:150]
        self.source_len = 512
        self.summ_len = 128
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        ctext = str(self.X[index])
        ctext = ' '.join(ctext.split())

        text = str(self.y[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len,
                                                  pad_to_max_length=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length=self.summ_len,
                                                  pad_to_max_length=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        target_ids = target['input_ids'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long)
        }


def prepare_data_for_training():
    train_data = ReviewDataset('data/train.csv')
    test_data = ReviewDataset('data/test.csv')

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=4)
    return train_loader, test_loader


