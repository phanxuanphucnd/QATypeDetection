# -*- coding: utf-8 -*
# Copyright (c) 2021 by Phuc Phan - Onion

import torch
import numpy as np


class QATypeDataset(torch.utils.data.Dataset):

    def __init__(self, data_df, tokenizer, label_dict):
        self.labels = [label_dict[label] for label in data_df['label']]
        self.texts = [
            tokenizer(text, padding='max_length', max_length = 512, 
            truncation=True, return_tensors="pt") for text in data_df['text']
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
