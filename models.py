#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
FileName: models.py
Description:
Author: Stark Lv
Date: 2021/2/27 6:13 PM
Version: 0.1
"""

from transformers import BertModel, BertTokenizer
from torch import nn
import torch
import torch.nn.functional as F


class BertBaseLinear(nn.Module):
    """bert-base-chinese with Linear output"""

    def __init__(self, model_path):
        super().__init__()
        self.bert_base = BertModel.from_pretrained(model_path)
        self.ocnli_linear = nn.Linear(768, 2)

    def forward(self, *inputs):
        """
        bert的输出.pooled_output([1],bacth_size * hidden_size),是last_hidden_state的第一个
        token[CLS]接了一个线性层(hidden_size * hidden_size)在经过Tanh激活函数得到的
        并不直接等于last_hidden_state([0], batch_size * sequence_len * hidden_size)的
        第一个token[CLS],所以我们一般还要进一步的微调,就会使用last_hidden_state,而不直接使用
        pooled_output
        """
        cls_embs = self.bert_base(*inputs)[0][:, 0, :].squeeze(1)
        return self.ocnli_linear(cls_embs)


class ChineseRobertaLinear(nn.Module):
    '''hfl/chinese-roberta-wwm-ext with linear output'''
    def __init__(self, model_path):
        super().__init__()
        self.bert_base = BertModel.from_pretrained(model_path)
        self.ocnli_linear = nn.Linear(768, 2)

    def forward(self, *inputs):
        """
        bert的输出.pooled_output([1],bacth_size * hidden_size),是last_hidden_state的第一个
        token[CLS]接了一个线性层(hidden_size * hidden_size)在经过Tanh激活函数得到的
        并不直接等于last_hidden_state([0], batch_size * sequence_len * hidden_size)的
        第一个token[CLS],所以我们一般还要进一步的微调,就会使用last_hidden_state,而不直接使用
        pooled_output
        """
        cls_embs = self.bert_base(*inputs)[0][:, 0, :].squeeze(1)
        return self.ocnli_linear(cls_embs)


class BertBaseAttention(nn.Module):
    '''hfl/chinese-roberta-wwm-ext with linear output
    Chinese-roberta-base-chinese with self-attention output
    The implementated self-attention mechamism is same as
    'A Structured Self-attentive Sentence Embedding' in ICLR2017
    without penalization item.
    '''

    def __init__(self, model_path):
        super().__init__()
        self.roberta_base = BertModel.from_pretrained(model_path)

        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        da = 350
        r = 30
        self.W_s1 = nn.Linear(768, da)
        self.W_s2 = nn.Linear(da, r)
        self.ocnli_linear = nn.Linear(r * 768, 2)

    def forward(self, *inputs):
        outputs = self.roberta_base(*inputs)[0]
        hidden_matrix = self.self_att(outputs)
        cls_embs = self.ocnli_linear(hidden_matrix.view(hidden_matrix.size(0), -1))
        return cls_embs

    def self_att(self, embs):
        attention_matrix = self.W_s2(torch.tanh(self.W_s1(embs)))
        attention_matrix = F.softmax(attention_matrix.permute(0, 2, 1), dim=2)
        hidden_matrix = torch.bmm(attention_matrix, embs)
        return hidden_matrix

class ChineseRobertaAttention(nn.Module):
    '''hfl/chinese-roberta-wwm-ext with linear output
    Chinese-roberta-base-chinese with self-attention output
    The implementated self-attention mechamism is same as
    'A Structured Self-attentive Sentence Embedding' in ICLR2017
    without penalization item.
    '''

    def __init__(self, model_path):
        super().__init__()
        self.roberta_base = BertModel.from_pretrained(model_path)
        #self.dropout = nn.Dropout(0.2)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        da = 350
        r = 30
        self.W_s1 = nn.Linear(768, da)
        self.W_s2 = nn.Linear(da, r)
        self.ocnli_linear = nn.Linear(r * 768, 2)

    def forward(self, *inputs):
        outputs = self.roberta_base(*inputs)[0]
        hidden_matrix = self.self_att(outputs)
        cls_embs = self.ocnli_linear(hidden_matrix.view(hidden_matrix.size(0), -1))
        return cls_embs

    def self_att(self, embs):
        attention_matrix = self.W_s2(torch.tanh(self.W_s1(embs)))
        attention_matrix = F.softmax(attention_matrix.permute(0, 2, 1), dim=2)
        hidden_matrix = torch.bmm(attention_matrix, embs)
        return hidden_matrix




##所有模型
MODELS = {
    "BertBaseLinear": {
        "class": BertBaseLinear,
        "tokenizer": BertTokenizer,
        "path": "./pretrain_model/bert_base_chinese/"
    },
    "ChineseRobertaAttention": {
        "class": ChineseRobertaAttention,
        "tokenizer": BertTokenizer,
        "path": "./pretrain_model/chinese_roberta_wwm_ext/"
    },
    "ChineseRobertaLinear": {
        "class": ChineseRobertaLinear,
        "tokenizer": BertTokenizer,
        "path": "./pretrain_model/chinese_roberta_wwm_ext/"
    },
    "BertBaseAttention": {
        "class": BertBaseAttention,
        "tokenizer": BertTokenizer,
        "path": "./pretrain_model/bert_base_chinese/"
    }
}
