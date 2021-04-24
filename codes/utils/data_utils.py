#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: bert_pipeline.py.py

@time: 2021/02/01 10:30

@desc: 数据读取代码

"""
import torch
import numpy as np
from torch.utils.data import Dataset

from codes.utils.process import truncate_seq_pair, truncate_seq, padding
from codes.config import config_


class MlmData(Dataset):
    def __init__(self, data, tokens_map, max_seq_len=32, concat=True, random=True):
        """
        args:
        data: list, [(text_a, text_b, label), ...]
        tokens_map: dict, {'a': 1, 'b': 2}
        max_seq_len: int
        concat: bool
        random: bool, 是否随机mask
        """
        self.special_tokes = {'pad': 0, 'unk': 1, 'cls': 2, 'sep': 3, 'mask': 4, 'no': 5, 'yes': 6}
        self.tokens_map = tokens_map
        self.random = random
        self.max_seq_len = max_seq_len
        self.concat = concat

        self.texts_a, self.texts_b, self.labels = zip(*data)

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, idx):
        text_ids_a = [self.tokens_map.get(t, self.special_tokes['unk']) for t in self.texts_a[idx]]
        text_ids_b = [self.tokens_map.get(t, self.special_tokes['unk']) for t in self.texts_b[idx]]
        # 截断 text_ids_a 和 text_ids_b
        text_ids_a, text_ids_b = truncate_seq_pair(text_ids_a, text_ids_b, self.max_seq_len * 2)

        if self.random:
            if np.random.random() < 0.5:
                text_ids_a, text_ids_b = text_ids_b, text_ids_a
            text_ids_a, out_ids_a = self.random_mask(text_ids_a)
            text_ids_b, out_ids_b = self.random_mask(text_ids_b)

        else:
            out_ids_a = [0] * len(text_ids_a)
            out_ids_b = [0] * len(text_ids_b)

        # 如果是要把两个句子合并在一起
        if self.concat:
            input_ids = [self.special_tokes['cls']] + text_ids_a + [self.special_tokes['sep']] + text_ids_b + \
                        [self.special_tokes['sep']]
            token_type_ids = [0] * (len(text_ids_a) + 2) + [1] * (len(text_ids_b) + 1)
            attention_mask = [1] * (len(text_ids_a) + len(text_ids_b) + 3)

            # 补全长度
            input_ids = padding(input_ids, self.max_seq_len * 2, self.special_tokes['pad'])
            token_type_ids = padding(token_type_ids, self.max_seq_len * 2, self.special_tokes['pad'])
            attention_mask = padding(attention_mask, self.max_seq_len * 2, self.special_tokes['pad'])

            out_ids = [self.special_tokes['cls']] + out_ids_a + [self.special_tokes['sep']] + out_ids_b + \
                      [self.special_tokes['sep']]
            out_ids = padding(out_ids, self.max_seq_len * 2, self.special_tokes['pad'])

            assert len(input_ids) == len(out_ids) == len(token_type_ids) == len(attention_mask)

            input_ids = torch.tensor(input_ids)
            token_type_ids = torch.tensor(token_type_ids)
            attention_mask = torch.tensor(attention_mask)
            out_ids = torch.tensor(out_ids)
            label = torch.tensor(self.labels[idx])

            return input_ids, token_type_ids, attention_mask, out_ids, label

        else:
            # TODO: 不进行concat，单条数据输入bert
            text_a = truncate_seq(text_ids_a, self.max_seq_len)
            text_b = truncate_seq(text_ids_b, self.max_seq_len)
            text_a = padding(text_a, self.max_seq_len, pad=self.special_tokes['pad'])
            text_b = padding(text_b, self.max_seq_len, pad=self.special_tokes['pad'])

    def random_mask(self, text_ids):
        """
        随机mask
        """
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(4)
                output_ids.append(i)
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)
            elif r < 0.15:
                input_ids.append(np.random.choice(len(self.tokens_map)) + 7)
                output_ids.append(i)
            else:
                input_ids.append(i)
                output_ids.append(0)

        return input_ids, output_ids


if __name__ == '__main__':
    mlm_d = MlmData(config_.get_dataset_path('oppo_breeno_round1_data', 'train'))
