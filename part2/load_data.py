import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
TOKENIZER_NAME = 'google-t5/t5-small'
BOS_TOKEN = '<extra_id_0>'
LT_TOKEN = 'LESS_THAN'  # T5 vocab has no '<'


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_NAME)
        self.bos_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        self.encoder_ids, self.decoder_ids = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_lines = load_lines(os.path.join(data_folder, f'{split}.nl'))
        encoder_ids = [tokenizer(x).input_ids for x in nl_lines]
        if split == 'test':
            return encoder_ids, None
        sql_lines = load_lines(os.path.join(data_folder, f'{split}.sql'))
        decoder_ids = [tokenizer(y.replace('<', LT_TOKEN)).input_ids for y in sql_lines]
        return encoder_ids, decoder_ids

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        enc = torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        bos = torch.tensor([self.bos_id], dtype=torch.long)
        if self.decoder_ids is None:
            return enc, bos
        dec = torch.tensor(self.decoder_ids[idx], dtype=torch.long)
        return enc, dec, bos


def normal_collate_fn(batch):
    enc, dec, bos = zip(*batch)
    encoder_ids = pad_sequence(enc, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    full = [torch.cat([b, d]) for b, d in zip(bos, dec)]
    decoder_inputs = pad_sequence([x[:-1] for x in full], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([x[1:] for x in full], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(bos)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    enc, bos = zip(*batch)
    encoder_ids = pad_sequence(enc, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack(bos)
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
