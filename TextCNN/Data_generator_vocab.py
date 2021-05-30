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
from collections import Counter
from sklearn.model_selection import train_test_split
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np


def convert_to_bert_datasets(data, labels, max_length, v2i):
    input_ids = []

    for inputs in data:
        padded = np.full(max_length, fill_value=0, dtype=np.int32)
        padded[:len(inputs)] = [v2i[x] for x in inputs]
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
        self.unk = 1
        self.pad = 0

        t0 = time.time()
        print(f"{'Data_precessing':*^80}")
        train_data = pd.read_csv("../tianchi_datasets/track3_round1_newtrain3.tsv", sep="\t", header=None,
                                 names=["sentence1", "sentence2", "labels"])
        train_data["document"] = train_data["sentence1"].str.cat(train_data["sentence2"], sep = " [SEP] ")
        train_data["document"] = train_data["document"].map(lambda x:[word for word in x.split(" ")])
        test_data = pd.read_csv("../tianchi_datasets/track3_round1_testA.tsv", sep="\t", header=None,
                                names=["sentence1", "sentence2"])
        test_data["document"] = test_data["sentence1"].str.cat(test_data["sentence2"], sep=" [SEP] ")
        test_data["document"] = test_data["document"].map(lambda x: [word for word in x.split(" ")])
        print("## dataset size is {}".format(len(train_data)))

        # 计算最大长度
        self.max_len = self.cal_max_length(train_data, test_data)

        self.vocab, self.v2i, self.i2v = self.get_vocab_v2i_i2v(train_data, test_data)

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

        train = convert_to_bert_datasets(X_train, y_train, self.max_len, self.v2i)
        validation = convert_to_bert_datasets(X_val, y_val, self.max_len, self.v2i)
        test = convert_to_bert_datasets(test_data["document"], None, self.max_len, self.v2i)

        train_dataloader = Data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_dataloader = Data.DataLoader(validation, batch_size=batch_size, shuffle=False)
        test_dataloader = Data.DataLoader(test, batch_size=batch_size, shuffle=False)

        return train_dataloader, valid_dataloader, test_dataloader





    def get_vocab_v2i_i2v(self, train_data, test_data):
        ## 获得语料库vocab, id2word, word2id
        #min_count = 5
        word_counter = Counter()
        vocab = []
        for data in [train_data, test_data]:
            for text in data["document"]:
                for word in text:
                    word_counter[word] += 1

        for word, count in word_counter.most_common():
            vocab.append(word)

        v2i = {v:i for i,v in enumerate(vocab)}
        i2v = {i:v for v, i in v2i.items()}

        return vocab, v2i, i2v


    def cal_max_length(self,train_data, test_data):
        # 计算最大输入长度
        max_len = 0
        for data in [train_data, test_data]:
            for x in data["document"]:
                max_len = max(max_len, len(x)) ##这里加2是因为得开头和结尾加上[CLS]和[SEP]

        return max_len

    def get_loaders(self):
        return self.train_loader, self.valid_loader, self.test_loader

