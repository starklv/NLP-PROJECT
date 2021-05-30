#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
FileName: Data_generator.py
Description: 生成数据loader并预处理
Author: Stark Lv
Date: 2021/2/27 16:32 PM
Version: 0.1
"""



import torch.utils.data as Data
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np


def convert_to_bert_datasets(data, labels, max_length):
    input_ids = []

    for x in data:
        padded = np.full(max_length, fill_value=0, dtype=np.int32)
        padded[:len(x)] = x
        # embedding层对于输入有严格的要求，必须是torch.long类型，即int64类型
        input_ids.append(torch.LongTensor([padded]))

    input_ids = torch.cat(input_ids, dim=0)

    if labels is None:
        return Data.TensorDataset(input_ids)
    else:
        labels = torch.tensor(labels.values)
        return Data.TensorDataset(input_ids, labels)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds = elapsed_rounded))

class Corpus():

    def __init__(self, batch_size, seed_val):

        self.batch_size = batch_size
        self.seed_val = seed_val
        # 0.1表示10折交叉验证,训练集和验证集比例
        self.test_scale = 0.1

        t0 = time.time()
        print(f"{'Data_precessing':*^80}")
        train_data = pd.read_csv("./tianchi_datasets/track3_round1_newtrain.tsv", sep="\t", header=None,
                                 names=["sentence1", "sentence2", "labels"])
        test_data = pd.read_csv("./tianchi_datasets/track3_round1_testA.tsv", sep="\t", header=None,
                                names=["sentence1", "sentence2"])

        print("## dataset size is {}".format(len(train_data)))
        ## 将以空格符为间隔的字符串转化为数字形式的列表
        self.convrt_str_to_list(train_data, test_data)
        ## 获得语料库vocab, idx2label(并没有v2i的作用,只是为了确定'[CLS]', '[SEP]'的id),
        # label2idx(并没有i2v的作用), 几乎把原来的数字全部
        self.vocab, self.v2i, self.i2v = self.get_vocab_v2i_i2v(train_data, test_data)
        # 计算最大长度
        self.max_len = self.cal_max_length(train_data, test_data)

        train_data["document"] = [[self.v2i['[CLS]']] + train_data["sentence1"][i] + [self.v2i['[SEP]']] +
                                  train_data["sentence2"][i] + [self.v2i['[SEP]']] for i in range(len(train_data))]
        test_data["document"] = [[self.v2i['[CLS]']] + test_data["sentence1"][i] + [self.v2i['[SEP]']] +
                                 test_data["sentence2"][i] + [self.v2i['[SEP]']] for i in range(len(test_data))]

        print(f"{'Load  dataset ....':*^80}")
        self.train_loader, self.valid_loader, self.test_loader \
        = self.load_generator(self.test_scale,train_data[["document", "labels"]], test_data[["document"]],
            self.batch_size)

        load_time = format_time(time.time() - t0)
        print("## Load Datasets Consume {} s ###".format(load_time))
        print(f"{'All Train and Test Data loaded !':*^80}")

    def load_generator(self, test_size, train_data, test_data, batch_size):

        # 划分训练集和验证集,同时根据label值分层抽样
        X_train, X_val, y_train, y_val = train_test_split(
            train_data["document"], train_data["labels"], test_size=test_size,
            stratify=train_data["labels"], random_state=self.seed_val)

        train = convert_to_bert_datasets(X_train, y_train, self.max_len)
        validation = convert_to_bert_datasets(X_val, y_val, self.max_len)
        test = convert_to_bert_datasets(test_data["document"], None, self.max_len)

        train_dataloader = Data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_dataloader = Data.DataLoader(validation, batch_size=batch_size, shuffle=False)
        test_dataloader = Data.DataLoader(test, batch_size=batch_size, shuffle=False)

        return train_dataloader, valid_dataloader, test_dataloader

    def convrt_str_to_list(self, train_data, test_data):

        for n in  ["sentence1", "sentence2"]:
                for m in [train_data, test_data]:
                    m[n] = m[n].map(lambda x: [int(x) for x in x.split(" ")])



    def get_vocab_v2i_i2v(self, train_data, test_data):
        ## 获得语料库vocab, idx2label, label2idx
        vocab = set()
        for data in [train_data, test_data]:
            for n in ["sentence1", "sentence2"]:
                for i in range(len(data[n])):
                    self.regular(data[n][i])
                    vocab.update(data[n][i])



        v2i = {v: i for i, v in enumerate(sorted(vocab))}
        v2i['[PAD]'] = v2i.pop(0)
        v2i['[CLS]'] = len(v2i)
        v2i['[SEP]'] = len(v2i)
        i2v = {i: v for v, i in v2i.items()}

        return vocab, v2i, i2v


    def cal_max_length(self, train_data, test_data):
        # 计算最大输入长度
        max_len = max([len(s1) + len(s2) + 3 for s1, s2 in zip(
            train_data["sentence1"].tolist() + test_data["sentence1"].tolist(),
            train_data["sentence2"].tolist() + test_data["sentence2"].tolist())])

        return max_len

    def get_loaders(self):
        return self.train_loader, self.valid_loader, self.test_loader


    def regular(self, data):
        ## 因为bert的vocab size最大为21128,因此将大于21128的全部替换为0，['PAD'] 标记为0
        for i in range(len(data)):
            num = data[i]
            data[i] = num if num < 21125 else 0