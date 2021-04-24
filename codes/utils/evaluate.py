#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: evaluate.py

@time: 2021/02/01 10:30

@desc: 评估函数

"""
from sklearn import metrics


def metrics_auc(y, pred):
    """
    y:
    pred:
    :return:
    float
    """
    return metrics.roc_auc_score(y, pred)
