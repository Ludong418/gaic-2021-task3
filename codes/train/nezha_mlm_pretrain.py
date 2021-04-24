#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: bert_pipeline.py.py

@time: 2021/02/01 10:30

@desc: Using nezha-pretrained to pretrain from scratch

"""
import os
import json
import argparse

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW

from codes.models.modeling_nezha import NeZhaForMaskedLM
from codes.models.virtual_alum import virtual_adversarial_training
from codes.config import config_
from codes.utils.process import int_text_label, count_tokens, load_vocab_dict, write_vocab_dict
from codes.utils.data_utils import MlmData


parser = argparse.ArgumentParser(description='相似度')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='epochs')
parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--pretrained_model', default='nezha-large-zh', type=str, metavar='N',
                    help='预训练模型的名称')
parser.add_argument('--output_model', default='nezha-gaic-large-adv', type=str, metavar='N',
                    help='输出保存的模型的名称')
parser.add_argument('--token_min_num', default=5, type=int, metavar='N',
                    help='训练集和测试集的某个token的最小数量，如果小于则就去除')
parser.add_argument('--max_seq_len', default=16, type=int, metavar='N',
                    help='单条文本的最大长度')
parser.add_argument('--lr', default=1e-5, type=float, metavar='N',
                    help='学习率')
parser.add_argument('--keep_embeddings_method', default=2, type=int, metavar='N',
                    help='resize embeddings matrix 的方法，0：不resize，1：随机resize，2：指定 tokens resize')
parser.add_argument('--use_adv', default='vat', type=str, metavar='N',
                    help='是否使用对抗训练, vat、fgm、vat-fgm')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# mlm_model = NeZhaForMaskedLM.from_pretrained(config_.get_pretrained_path(args.pretrained_model))


def main():
    # 读取训练集、验证集、测试集
    train_f = open(config_.get_dataset_path('oppo_breeno_round1_data', 'train-expand'), 'r', encoding='utf8')
    test_f = open(config_.get_dataset_path('oppo_breeno_round1_data', 'test'), 'r', encoding='utf8')
    data = list(map(lambda x: int_text_label(x, '\t'), train_f.readlines()))
    train_data = [d for i, d in enumerate(data) if i % 10 != 0]
    valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
    test_data = list(map(lambda x: int_text_label(x, '\t', -5), test_f.readlines()))

    all_data = train_data + valid_data + test_data
    count = count_tokens(all_data, args.token_min_num)
    count = sorted(count.items(), key=lambda s: -s[1])
    # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes
    tokens_map = {t[0]: i + 7 for i, t in enumerate(count)}
    # 把词典写入文件
    vocab = list(zip(*sorted(tokens_map.items(), key=lambda item: item[1])))[0]
    vocab = ['pad', 'unk', 'cls', 'sep', 'mask', 'no', 'yes'] + list(vocab)
    write_vocab_dict(vocab, config_.get_pretrained_path(args.output_model, 'vocab.txt'))
    print('vocab 数量：', len(tokens_map) + 7)

    # BERT词频
    counts = json.load(open(config_.BERT_COUNT_PATH))
    del counts['[CLS]']
    del counts['[SEP]']
    token_dict = load_vocab_dict(config_.get_pretrained_path(args.pretrained_model, 'vocab.txt'))
    freqs = [counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])]
    keep_tokens = list(np.argsort(freqs)[::-1])

    # 把验证集和测试集加到训练集当中
    for d in valid_data + test_data:
        train_data.append((d[0], d[1], -5))

    train_dataset = MlmData(train_data, tokens_map, max_seq_len=args.max_seq_len, concat=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = NeZhaForMaskedLM.from_pretrained(config_.get_pretrained_path(args.pretrained_model))
    model.resize_token_embeddings(len(tokens_map) + 7)

    # 模型指定GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义优化器和损失函数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.00001},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        loss = train(train_loader, model, optimizer, device, step)
        print('loss: %s' % loss)
        model.save_pretrained(config_.get_pretrained_path(args.output_model))


def train(train_loader, model, optimizer, device, step):
    model.train()
    train_loss = 0

    for input_ids, token_type_ids, attention_mask, output_ids, _ in tqdm(train_loader):
        step += 1
        input_ids = input_ids.long().to(device)
        token_type_ids = token_type_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)
        output_ids = output_ids.long().to(device)
        optimizer.zero_grad()
        # 混合精度计算，训练速度接近提高了1/2
        with autocast():
            output = model(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           labels=output_ids)
            loss = output[0]
            if args.use_adv == 'vat':
                logits = output[1]
                hidden_status = output[2][0]
                adv_loss = virtual_adversarial_training(model, hidden_status, token_type_ids, attention_mask, logits)
                if adv_loss:
                    loss = adv_loss * 10 + loss

        train_loss += loss
        loss.backward()
        optimizer.step()

    return train_loss / len(train_loader)


if __name__ == '__main__':
    main()
