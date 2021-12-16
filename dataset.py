from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
import pandas as pd
import json
from functools import reduce
from operator import add


class BiRdQA(Dataset):
    def __init__(self, riddles_path, info_path, device, n_options=5, max_length=64):
        self.tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
        self.n_options = n_options
        self.device = device
        df = pd.read_csv(riddles_path)
        riddles = self.tokenizer(df.riddle.to_list(),
                                 padding=True,
                                 truncation=True,
                                 max_length=max_length,
                                 return_tensors='pt')
        self.riddle_input_ids = riddles['input_ids'].to(device)
        self.riddle_attention_mask = riddles['attention_mask'].to(device)

        with open(info_path, 'r') as f:
            info = json.load(f)
        length = len(df['choice0'].to_list())
        all_options = reduce(add, (df[f'choice{i}'].to_list() for i in range(5)))
        all_options = [info[i] for i in all_options]
        all_options_tokenized = self.tokenizer(all_options,
                                               padding=True,
                                               truncation=True,
                                               max_length=max_length,
                                               return_tensors='pt')
        self.options_input_ids = torch.stack(torch.split(all_options_tokenized['input_ids'], length)).permute(1, 0, 2).contiguous().to(device)
        self.options_attention_mask = torch.stack(torch.split(all_options_tokenized['attention_mask'], length)).permute(1, 0, 2).contiguous().to(device)

        self.labels = torch.LongTensor(df.label.to_list()).to(device)

    def __len__(self):
        return self.riddle_input_ids.shape[0]

    def __getitem__(self, idx):
        riddle_input_ids = self.riddle_input_ids[idx]
        riddle_attention_mask = self.riddle_attention_mask[idx]
        options_input_ids = self.options_input_ids[idx]
        options_attention_mask = self.options_attention_mask[idx]
        label = self.labels[idx]
        return (riddle_input_ids,
                riddle_attention_mask,
                options_input_ids,
                options_attention_mask,
                label)
