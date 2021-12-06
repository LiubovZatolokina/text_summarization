import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

from dataset import prepare_data_for_training

tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

train_loader, test_loader = prepare_data_for_training()

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

learning_rate = 1e-4
num_epochs = 300
model_saving_path = './t5_model.pt'


def train_model(model, dataloaders, optimizer, num_epochs, model_saving_path):
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    since = time.time()
    tb = SummaryWriter()
    for epoch in tqdm(range(num_epochs)):
        loss_dict = {}
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    source = tokenizer.batch_encode_plus(np.array(inputs), max_length=512, padding=True,
                                                         pad_to_max_length=True, return_tensors='pt', truncation=True)
                    target = tokenizer.batch_encode_plus(np.array(labels), max_length=128, pad_to_max_length=True,
                                                         return_tensors='pt', truncation=True, padding=True)
                    source.to(device)
                    target.to(device)

                    source_ids = source['input_ids'].squeeze()
                    source_mask = source['attention_mask'].squeeze()
                    target_ids = target['input_ids'].squeeze()
                    # y_ids = target_ids[:, :-1].contiguous()
                    # lm_labels = target_ids[:, 1:].clone().detach()
                    # lm_labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100
                    #target_mask = target['attention_mask'].squeeze()
                    outputs = model(input_ids=source_ids.to(device), attention_mask=source_mask.to(device),
                                    decoder_input_ids=target_ids.to(device))
                    loss = outputs[0]
                    running_loss += loss.item() * len(inputs[0])
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(epoch_loss)
            loss_dict[phase] = epoch_loss
            model.save_pretrained("t5_model")
            torch.save(model.state_dict(), model_saving_path)

        tb.add_scalars('Loss: epoch', {'Train': loss_dict['train'], 'Valid': loss_dict['valid']}, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    torch.cuda.empty_cache()
    t5_model.to(device)
    optimizer = torch.optim.Adam(t5_model.parameters(), lr=learning_rate)
    dataloaders_dict = {'train': train_loader, 'valid': test_loader}
    model_ft = train_model(t5_model, dataloaders_dict, optimizer, num_epochs, model_saving_path)