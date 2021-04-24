#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: bert_pipeline.py.py

@time: 2021/02/01 10:30

@desc: Using nezha-pretrained to find-tuning

"""
import os
import argparse

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from codes.models.modeling_nezha import NeZhaForSequenceClassification
from codes.models.virtual_alum import virtual_adversarial_training
from codes.config import config_
from codes.utils.process import int_text_label, load_vocab_dict
from codes.utils.data_utils import MlmData


parser = argparse.ArgumentParser(description='分类模型')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='epochs')
parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--pretrained_model', default='nezha-gaic-large', type=str, metavar='N',
                    help='预训练模型的名称')
parser.add_argument('--output_model', default='nezha-gaic-semantic-large-expand', type=str, metavar='N',
                    help='输出保存的模型的名称')
parser.add_argument('--max_seq_len', default=16, type=int, metavar='N',
                    help='单条文本的最大长度')
parser.add_argument('--lr', default=1e-5, type=float, metavar='N',
                    help='学习率')
parser.add_argument('--use_adv', default='vat', type=str, metavar='N',
                    help='是否使用对抗训练, vat、fgm、vat-fgm')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # 读取训练集、验证集、测试集
    train_f = open(config_.get_dataset_path('oppo_breeno_round1_data', 'train-expand'), 'r', encoding='utf8')
    data = list(map(lambda x: int_text_label(x, '\t'), train_f.readlines()))
    train_data = [d for i, d in enumerate(data) if i % 10 != 0]
    valid_data = [d for i, d in enumerate(data) if i % 10 == 0]

    token_dict = load_vocab_dict(config_.get_pretrained_path(args.pretrained_model, 'vocab.txt'))
    tokens_map = {t: i for i, t in enumerate(token_dict)}
    print('vocab 数量：', len(tokens_map))

    train_dataset = MlmData(train_data, tokens_map, max_seq_len=args.max_seq_len, concat=True, random=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataset = MlmData(valid_data, tokens_map, max_seq_len=args.max_seq_len, concat=True, random=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    model = NeZhaForSequenceClassification.from_pretrained(config_.get_pretrained_path(args.pretrained_model))
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
        print('train-loss: %s' % loss)
        model.save_pretrained(config_.get_pretrained_path(args.output_model))
        eval_loss, auc, acc = evaluate(dev_loader, model, device)
        print('eval-loss: %s, auc: %s, acc: %s' % (eval_loss, auc, acc))


def train(train_loader, model, optimizer, device, step):
    model.train()
    train_loss = 0

    for input_ids, token_type_ids, attention_mask, _, labels in tqdm(train_loader):
        step += 1
        input_ids = input_ids.long().to(device)
        token_type_ids = token_type_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)
        labels = labels.long().to(device)
        optimizer.zero_grad()
        with autocast():
            output = model(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           labels=labels)

            loss = output[0]
            if args.use_adv == 'vat':
                logits = output[1]
                hidden_status = output[2][0]
                adv_loss = virtual_adversarial_training(model, hidden_status, token_type_ids, attention_mask, logits)
                if adv_loss:
                    loss = adv_loss + loss

        train_loss += loss
        loss.backward()
        optimizer.step()

    return train_loss / len(train_loader)


def evaluate(data_loader, model, device):
    model.eval()
    eval_loss = 0
    eval_acc = 0
    y_true = []
    y_predict = []

    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, _, labels in data_loader:
            input_ids = input_ids.long().to(device)
            token_type_ids = token_type_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            labels = labels.long().to(device)

            output = model(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           labels=labels)

            loss = output[0]
            logits = output[1]
            acc = ((logits.argmax(dim=-1) == labels).sum()).item()
            eval_acc += acc / logits.shape[0]

            predict_scores = logits.softmax(-1)
            # 预测为1的概率值
            predict_scores = predict_scores[:, 1]
            y_predict.extend(predict_scores.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            eval_loss += loss

    return eval_loss / len(data_loader), roc_auc_score(y_true, y_predict), eval_acc / len(data_loader)


if __name__ == '__main__':
    main()