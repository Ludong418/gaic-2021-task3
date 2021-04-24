#!/usr/bin/python

# encoding: utf-8

"""
@author: ludong

@contact: ludong@cetccity.com

@software: PyCharm

@file: base.py

@time: 2021/02/01 10:30

@desc: 配置文件
"""
import os
import yaml

yaml.warnings({'YAMLLoadWarning': False})


class YamlConfig(object):
    def __init__(self, yml_file_path='config.yml'):
        # config 文件夹路径
        self.CUR_PATH = os.path.abspath(os.path.dirname(__file__))
        # 项目路径
        self.PROJECT_PATH = os.path.abspath(os.path.join(self.CUR_PATH, "../train"))
        self._yml_file = os.path.join(self.CUR_PATH, yml_file_path)
        with open(self._yml_file, 'r', encoding='utf-8') as f:
            self._yml_obj = yaml.load(f)
        # 数据集的路径
        self.DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(self.PROJECT_PATH)), 'tcdata')
        # 预训练模型的文件夹路径
        self.PRETRAINED_MODEL_PATH = self._yml_obj['pretrained']
        # bert count 文件
        self.BERT_COUNT_PATH = os.path.join(self.CUR_PATH, self._yml_obj['bert_count'])
        # model logs 文件夹
        self.MODEL_LOGS = os.path.join(os.path.dirname(os.path.dirname(self.PROJECT_PATH)), 'model_logs')
        # prediction result 预测结果
        self.PREDICTION_RESULT = os.path.join(os.path.dirname(os.path.dirname(self.PROJECT_PATH)),
                                              'prediction_result',
                                              'result.tsv')


class ModelConfig(YamlConfig):
    def __init__(self):
        super(ModelConfig, self).__init__()
        self.DATASET_PATH_DICT = self.generate_dataset_path()

    def generate_dataset_path(self):
        """
        生成数据集的绝对路径
        :return:
        """
        dataset_files = self._yml_obj['dataset']
        dataset_paths = {}
        for k, v in dataset_files.items():
            dataset_paths[k] = {}
            for mode, name in v.items():
                dataset_paths[k][mode] = os.path.join(self.DATASET_PATH, k, name)

        return dataset_paths

    def get_dataset_path(self, name, mode):
        """
        获取数据集的路径
        name: str, 数据集的目录名称
        mode: str, 'train'、'dev'、'test'
        :return:
        """
        return self.DATASET_PATH_DICT[name][mode]

    def get_pretrained_path(self, name, file=None):
        if file:
            return os.path.join(self.PRETRAINED_MODEL_PATH, name, file)
            # return self.PRETRAINED_MODEL_PATH + '/' + name + '/' + file
        else:
            return os.path.join(self.PRETRAINED_MODEL_PATH, name)


if __name__ == '__main__':
    mc = ModelConfig()
    print(mc.DATASET_PATH_DICT)
