# -*- coding: utf-8 -*
# Copyright (c) 2021 by Phuc Phan - Onion

import os
import torch
import numpy as np
import pandas as pd

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from transformers import BertTokenizer

from model import BertClassifier
from dataset import QATypeDataset

np.random.seed(112)

class QATypeLeaner():
    def __init__(self, model_path=None, pretrained_model="bert-base-uncased", device=None) -> None:
        self.model_path = model_path
        self.pretrained_model = pretrained_model
        self.label_dict = {}
        
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f">>> Using device: {self.device}")
        
        self.model = BertClassifier()
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
        if self.model_path:
            self.load(model_path=self.model_path)

    def load(self, model_path):
        model_path = os.path.abspath(model_path)
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.label_dict = self.checkpoint['label_dict']

        print(f"Label dictionary: {self.label_dict}.")
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    
    def train(self, train_df, val_df, batch_size=8, learning_rate=1e-5, epochs=10):
        
        self.label_dict = {}
        
        labels = train_df['label'].unique()
        for index, label in enumerate(labels):
            self.label_dict[label] = index

        print(f"Label dictionary: {self.label_dict}.")

        train_dataset = QATypeDataset(train_df, self.tokenizer, self.label_dict)
        val_dataset = QATypeDataset(val_df, self.tokenizer, self.label_dict)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.model = self.model.to(self.device)
        criterion = criterion.to(self.device)

        for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(self.device)
                mask = train_input['attention_mask'].to(self.device)
                input_id = train_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():
                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(self.device)
                    mask = val_input['attention_mask'].to(self.device)
                    input_id = val_input['input_ids'].squeeze(1).to(self.device)

                    output = self.model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_df): .4f} "
                f"| Train Accuracy: {total_acc_train / len(train_df): .4f} "
                f"| Val Loss: {total_loss_val / len(val_df): .4f} "
                f"| Val Accuracy: {total_acc_val / len(val_df): .4f}")

        save_dir = './models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save({
                    'model_state_dict': self.model.state_dict(), 
                    'label_dict': self.label_dict, 
                    
                }, os.path.join(save_dir, 'final_model.model'))

        print(f"Saved model to `{os.path.join(save_dir, 'final_model.model')}.`")
    
    def evaluate(self, test_df, batch_size=16):
        test_dataset = QATypeDataset(test_df, self.tokenizer, self.label_dict)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        label_dict_inversed = {v: k for k, v in self.label_dict.items()}
        
        total_acc_test = 0

        preds = []

        with torch.no_grad():
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(self.device)
                mask = test_input['attention_mask'].to(self.device)
                input_id = test_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
        
        print(f'Test Accuracy: {total_acc_test / len(test_df): .4f}')

    def inference(self, input_text):
        if not self.model:
            raise ValueError(f"The model_path is  None or invalid.")
        
        label_dict_inversed = {v: k for k, v in self.label_dict.items()}
        input_text = input_text.lower()
        
        with torch.no_grad():
            input = self.tokenizer(input_text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            mask = input['attention_mask'].to(self.device)
            input_id = input['input_ids'].squeeze(1).to(self.device)
            prediction = self.model(input_id, mask)[0]

        score = torch.nn.functional.softmax(prediction).cpu().data.numpy()
        print("Score: ", score)
        max_index = np.argmax(score)

        return {
            'class': label_dict_inversed[max_index],
            'score': score[max_index]
        }

    def inference2(self, input_text):
        if not self.model:
            raise ValueError(f"The model_path is  None or invalid.")
        
        label_dict_inversed = {v: k for k, v in self.label_dict.items()}
        input_text = input_text.lower()

        data_df = pd.DataFrame([[input_text, 'AGREE']], columns=['text', 'label'])
        dataset = QATypeDataset(data_df, self.tokenizer, self.label_dict)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        total_acc_test = 0

        with torch.no_grad():
            for test_input, test_label in dataloader:
                test_label = test_label.to(self.device)
                mask = test_input['attention_mask'].to(self.device)
                input_id = test_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)

        score = torch.nn.functional.softmax(output).cpu().data.numpy()
        print()
        print("Score: ", score)
        max_index = np.argmax(score)

        return {
            'class': label_dict_inversed[max_index],
            'score': score[0][max_index]
        }
    