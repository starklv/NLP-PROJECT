#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
FileName: Data_generator.py
Description: 生成数据loader并预处理
Author: Stark Lv
Date: 2021/2/27 16:32 PM
Version: 0.1
"""

import json
import torch.utils.data as Data
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def convert_to_bert_dataset(data, labels, tokenizer, max_length, name):
    input_ids = []
    token_type_ids = []
    attention_masks = []

    for x in tqdm(data, desc="convert_to_bert_{}_dataset".format(name)):
        encoded_input = tokenizer.encode_plus(x, add_special_tokens=True,
                                              max_length=max_length, padding="max_length",
                                              return_attention_mask=True,
                                              return_tensors="pt", truncation=True)

        input_ids.append(encoded_input["input_ids"])
        attention_masks.append(encoded_input["attention_mask"])
        # 对于bert,输入有token_type_ids,其他模型没有
        try:
            token_type_ids.append(encoded_input["token_type_ids"])
        except:
            pass

    # convert lists to tensor, bert在Pytorch中只接受torch格式的输入
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if len(token_type_ids) != 0:
        token_type_ids = torch.cat(token_type_ids, dim=0)

    # 对于测试集，没有labels
    if labels is None:
        # 对于非bert类输入,没有token_type_ids
        if len(token_type_ids) == 0:
            return Data.TensorDataset(input_ids, attention_masks)
        return Data.TensorDataset(input_ids, attention_masks, token_type_ids)

    else:
        labels = torch.tensor(labels.values)
        if len(token_type_ids) == 0:
            return Data.TensorDataset(input_ids, attention_masks, labels)
        return Data.TensorDataset(input_ids, attention_masks, token_type_ids, labels)




class Corpus():

    def __init__(self, tokenizer, batch_size, seed_val):

        self.batch_size = batch_size
        self.seed_val = seed_val
        # 保持验证集和测试集数量大致相同,约为1500,1500/53386 = 0.0281
        # 0.1表示10折交叉验证,训练集和验证集比例
        self.test_scale = 0.1
        self.tokenizer = tokenizer
        with open("v2i_chinese.json", 'r') as fr:
            self.v2i_chinese = json.load(fr)

        print(f"{'Data_precessing':*^80}")
        """
        “当文本文件中带有英文双引号时，直接用pd.read_csv进行读取会导致行数减少，
        此时应该对read_csv设置参数quoting=3或者quoting=csv.QUOTE_NONE”
        不设置quoting，默认会去除英文双引号，只留下英文双引号内的内容，设置quoting = 3，会如实读取内容。
        """
        ## 这里的new_train是经过数据增强之后的效果:1.利用模型对test数据集进行预测,然后将分类概率大于0.95
        ## 的样本重新加入模型;2.使用了词向量方法提取测试集的向量与训练集的向量进行相似度计算，并且将较高
        ## 相似度的标签拿来做测试集的标签
        train_data = pd.read_csv("./tianchi_datasets/track3_round1_newtrain3.tsv", sep="\t", header=None,
                                  quoting=3, encoding="utf-8", names=["sentence1", "sentence2", "labels"])
        train_data["sentence1_cn"] = train_data["sentence1"].apply(self.convert_to_chinese)
        train_data["document"] = train_data["sentence1"].str.cat(train_data["sentence2"], sep=" [SEP] ")
        test_data = pd.read_csv("./tianchi_datasets/track3_round1_testA.tsv", sep="\t", header=None,
                                 quoting=3, encoding="utf-8", names=["sentence1", "sentence2"])
        test_data["document"] = test_data["sentence1"].str.cat(test_data["sentence2"], sep=" [SEP] ")

        print("## dataset size is {}".format(len(train_data)))

        # 计算max length, 计算耗时,直接保存结果，大概需要耗时30s
        self.max_length = 100
        """
        max_len = max(self.cal_max_length(ocnli_train), self.cal_max_length(ocnli_test))
        print("!!!### max token length of FineTune model : ", max_len)
        """

        print(f"{'Load OCNLI dataset ....':*^80}")
        self.train_loader, self.valid_loader, self.test_loader,\
            = self.load_generator(self.test_scale,
            train_data[["document", "labels"]], test_data[["document"]], self.batch_size)

        print(f"{'All Train and Test Data loaded !':*^80}")

    def load_generator(self, test_size, train_data, test_data, batch_size):
        """生成训练、验证、测试集合的dataloader"""


        # 划分训练集和验证集,同时根据label值分层抽样
        X_train, X_val, y_train, y_val = train_test_split(
            train_data["document"], train_data["labels"], test_size=test_size,
            stratify= train_data["labels"], random_state=self.seed_val)


        train = convert_to_bert_dataset(X_train, y_train, self.tokenizer, self.max_length, "train")
        validation = convert_to_bert_dataset(X_val, y_val, self.tokenizer, self.max_length, "valid")
        test = convert_to_bert_dataset(test_data["document"], None, self.tokenizer, self.max_length, "test")

        train_dataloader = Data.DataLoader(
            train, batch_size=batch_size, shuffle=True)
        valid_dataloader = Data.DataLoader(
            validation, batch_size=batch_size, shuffle=False)
        ##因为要逐条预测
        test_dataloader = Data.DataLoader(test, batch_size=batch_size, shuffle=False)

        return train_dataloader, valid_dataloader, test_dataloader



    def cal_max_length(self, data):
        # 计算最大输入长度
        max_len = 0
        for x in tqdm(data["document"]):
            max_len = max(max_len, len(self.tokenizer(x)["input_ids"]))
        return max_len

    def convert_to_chinese(self, text_ids, v2i_chinese):
        s = "".join([self.v2i_chinese.get(int(num), '[PAD]') for num in text_ids.split(" ")])
        return s


    def get_loaders(self):
        return self.train_loader, self.valid_loader, self.test_loader