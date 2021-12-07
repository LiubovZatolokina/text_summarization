import torch
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from train_t5 import test_loader

config_t5 = T5Config.from_json_file('t5_model/config.json')
tokenizer_t5 = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5_model/pytorch_model.bin', config=config_t5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    sources = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            source = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) \
                     for g in ids]
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) \
                     for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            predictions.extend(preds)
            sources.extend(source)
    return predictions, sources


if __name__ == '__main__':
    t5_model.to(device)
    predictions, sources = validate(tokenizer_t5, t5_model, device, test_loader)
    print(predictions)
    print(sources)
