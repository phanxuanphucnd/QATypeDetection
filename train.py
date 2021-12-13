# -*- coding: utf-8 -*
# Copyright (c) 2021 by Phuc Phan - Onion                                                                                                                                                   

import pandas as pd

from learner import QATypeLeaner

data_path = './data.csv'

data_df = pd.read_csv(data_path, encoding='utf-8')
data_df['text'] = data_df['text'].apply(lambda x: x.lower())

train_df, test_df = data_df, data_df

epochs = 15
lr = 1e-5
bs = 8

learner = QATypeLeaner(pretrained_model='bert-base-cased')
learner.train(train_df=train_df, val_df=test_df, batch_size=bs, learning_rate=lr, epochs=epochs)

learner.evaluate(test_df=test_df, batch_size=16)
