# -*- coding: utf-8 -*
# Copyright (c) 2021 by Phuc Phan - Onion

import timeit

from learner import QATypeLeaner

learner = QATypeLeaner(model_path='models/final_model.model', pretrained_model='bert-base-cased', device='cpu')

# start = timeit.default_timer()
# output = learner.inference(input_text='hi')
# print(f"Ouput: {output}")

# stop = timeit.default_timer()
# print('Time: ', stop - start)  

# output = learner.inference2(input_text='hi')
# print(f"Ouput: {output}")

import pandas as pd

data_path = 'data/data.csv'

data_df = pd.read_csv(data_path, encoding='utf-8')
data_df['text'] = data_df['text'].apply(lambda x: x.lower())

test_df = data_df
learner.evaluate(test_df=test_df, batch_size=8)