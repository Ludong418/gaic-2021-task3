#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: bert_pipeline.py.py

@time: 2021/02/01 10:30

@desc:

"""
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from codes.config import config_
from codes.models.modeling_nezha import NeZhaForSequenceClassification
from codes.utils.process import int_text_label, load_vocab_dict
from codes.utils.data_utils import MlmData


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_name = 'nezha-gaic-semantic-large-expand'

test_f = open(config_.get_dataset_path('oppo_breeno_round1_data', 'test'), 'r', encoding='utf8')
test_data = list(map(lambda x: int_text_label(x, '\t', -5), test_f.readlines()))

token_dict = load_vocab_dict(config_.get_pretrained_path(model_name, 'vocab.txt'))
tokens_map = {t: i for i, t in enumerate(token_dict)}
print('vocab 数量：', len(tokens_map))

test_dataset = MlmData(test_data, tokens_map, max_seq_len=16, concat=True, random=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = NeZhaForSequenceClassification.from_pretrained(config_.get_pretrained_path(model_name))
# 模型指定GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

y_predict = []
with torch.no_grad():
    for input_ids, token_type_ids, attention_mask, _, _ in tqdm(test_loader):
        input_ids = input_ids.long().to(device)
        token_type_ids = token_type_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)

        output = model(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask)

        logits = output[0]
        predict_scores = logits.softmax(-1)
        # 预测为1的概率值
        predict_scores = predict_scores[:, 1]
        y_predict.extend(predict_scores.cpu().numpy())

with open(config_.PREDICTION_RESULT, 'w', encoding='utf8') as f:
    for y in y_predict:
        f.write(str(y) + '\n')
