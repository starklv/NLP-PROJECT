#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
FileName: finetune.py
Description:
Author: Stark Lv
Date: 2021/2/27 6:13 PM
Version: 0.1
"""

from transformers import get_linear_schedule_with_warmup
from TEXTCNN_pytorch import TEXTCNN
import torch
import logging
from torch import nn
from Data_generator_vocab import Corpus
import time
import datetime
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os
import json
import pandas as pd
from sklearn.metrics import roc_auc_score
from adv_utils import FGM



def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds = elapsed_rounded))

class fintune():

    def __init__(self, model, corpus):

        self.set_device() #创建了self.device, self.n_gpu
        self.set_random_seed() #设置随机数种子,保证所有结果可以复现

        # 加载训练集、验证集、测试集
        print(f"{'加载训练集、验证集、测试集 Loading':*^80}")
        self.train_loader, self.valid_loader, self.test_loader = corpus.get_loaders()

        #加载model
        #self.model, self.tokenizer = self.load_model_tokenizer(model_name)

        self.model = model
        self.model_name = "TextCNN"
        # move model to GPU
        if self.n_gpu > 1:
            device_ids = [0, 1, 2, 3]
            self.model =torch.nn.DataParallel(self.model, device_ids= device_ids)
        self.model = self.model.to(self.device)

        #loss function & optimizer
        # 带权重的损失函数, 若positive_label为1的f1_score为 0.75，positive_label为0的f1_score为0.85
        #self.loss_func = nn.CrossEntropyLoss(weight= torch.tensor([1.1765, 1.3333])).to(self.device)
        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 0.5])).to(self.device)
        self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr = HYPERS["LR"],
                eps = 1e-8)


    """
    def load_model_tokenizer(self, model_name):
        #model_path = os.path.join(os.getcwd(), MODELS[model_name]["path"])
        model_path = MODELS[model_name]["path"]
        assert model_name in MODELS

        if self.retrain_model_path is None:

            print(f"### 从 {model_path} 加载预训练模型, 重新训练 ###")
            model = MODELS[model_name]["class"](model_path)
        else:
            print(f"{'加载已预训练好的模型,继续训练':*^80}")
            model = torch.load("./fintuned_model/" + self.retrain_model_path)

        print(f"## Model {self.model_name} loaded. ##")

        tokenizer = MODELS[model_name]["tokenizer"].from_pretrained(model_path)

        ## __class__.__name__输出类的名字
        print(f"## Tokenizer {tokenizer.__class__.__name__} loaded. ##")

        return model, tokenizer
    """
    
    def set_random_seed(self):
        # Set the seed value all over the place to make this reproducible.
        self.seed_val = 2021
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)

    def set_device(self):
        ###设置日志时间的输出格式,说明在那段时间用过GPU
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        logging.basicConfig(filename='NLP_GPU.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

        # 设置运行环境GPU
        self.device = torch.device("cuda:0")
        self.n_gpu = torch.cuda.device_count()
        logging.info(msg = f"\n Using GPU, {self.n_gpu} device available")

    def run_epoch(self):

        #list to store a number of quantities such as
        #training and validation loss, validation accuracy, and timings.
        training_stats = []


        #Total number of training steps is [number of batches × number of epochs]
        total_steps = len(self.train_loader) * HYPERS["EPOCHS"]
        # create the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps = 100, num_training_steps = total_steps
        )

        total_t0 = time.time()

        print(f"### Trainng {HYPERS['EPOCHS']} EPOCHS start ###")
        for epoch in tqdm(range(1, HYPERS["EPOCHS"] + 1), unit = "epoch",
                          desc = f"Training All {HYPERS['EPOCHS']} Epochs"):
            # Measure how long the per training epoch takes
            t0 = time.time()

            # Put the model into training model
            print(f"### EPOCH {epoch} train  start")
            self.model.train()

            epoch_train_loss = self.train(epoch)
            # measure avg batch loss in specific epoch
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            print(f"## per batch train loss for epoch {epoch} is {avg_train_loss} ",
                  f"total train time : {training_time}")

            t0 = time.time()

            # eval mode
            # 在eval模式下不会应用DROPOUT和BATCHNORM
            print(f"### EPOCH {epoch} val start")
            self.model.eval()

            epoch_eval_loss, eval_auc = self.eval(metric=roc_auc_score)
            avg_eval_loss = epoch_eval_loss / len(self.valid_loader)
            validation_time = format_time(time.time() - t0)
            print(f"## per batch valid loss for epoch {epoch} is {avg_eval_loss} ",
                  f"total valid time : {validation_time}, Validation auc score {eval_auc}")

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'Epoch': epoch,
                    'Train Loss': epoch_train_loss,
                    'Avg Train Loss': avg_train_loss,
                    'Train Time': training_time,
                    'Valid Loss': epoch_eval_loss,
                    'Avg Valid Loss': avg_eval_loss,
                    'Valid Time': validation_time,
                    'Valid AUC': eval_auc,
                    'Test Results Dir': 'Model{}_BATCH{}_LR{}/'.format(self.model_name,
                     HYPERS['BATCH_SIZE'], HYPERS['LR'])
                }
            )

        test_t0 = time.time()
        self.test()
        print("Test Time Consume {}".format(format_time(time.time() - test_t0)))
        self.save_stats(training_stats)
        print(f"{'Training complete !':*^80}")
        print(f"### Total Process took {format_time(time.time() - total_t0)} ####")


    def train(self, epoch):
        total_train_loss = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}>>>>train", unit="batch"):
            # input_ids, attention_mask, token_type_id和Label混在一起
            inputs = [data.to(self.device) for data in batch[:-1]]
            labels = batch[-1].to(self.device)
            print(inputs[0].shape)
            outputs = self.model(*inputs)


            batch_loss = self.loss_func(outputs, labels)
            total_train_loss += batch_loss.to("cpu").data.numpy()
            # clear any previously calculated gradients before performing a backward pass.
            self.optimizer.zero_grad()

            # perform a backward pass to calculate the gradients
            batch_loss.backward()

            # normalization of the gradients to 1.0 to avoid exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters and take a step using the computed gradient
            self.optimizer.step()

            #update the learning rate 
            self.scheduler.step()

            #print(f"per batch loss {batch_loss.to('cpu').data.numpy()}")

        return total_train_loss

    def eval(self, metric):
        total_eval_loss = 0
        predicted_scores = np.array([])
        target_labels = np.array([])
        #predicted_labels = np.array([])

        for batch in self.valid_loader:
            inputs = [data.to(self.device) for data in batch[:-1]]
            labels = batch[-1].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                outputs = self.model(*inputs)
                loss = self.loss_func(outputs, labels)
                total_eval_loss += loss.to("cpu").data.numpy()

                # move logits and labels to CPU,存储在CPU上面的变量不能和存储在GPU上面的变量进行运算
                logits = F.softmax(outputs.to("cpu"), dim = 1).data.numpy()
                label_ids = labels.to("cpu").data.numpy()
                y_pred = logits[:, 1]
                #y_pred1 = np.argmax(logits, axis = 1)
                predicted_scores = np.append(predicted_scores, y_pred)
                #predicted_labels = np.append(predicted_labels, y_pred1)
                target_labels = np.append(target_labels, label_ids)


        task_auc = metric(target_labels, predicted_scores)

        #print(f"{'classification_report':*^80}")
        #print(classification_report(target_labels, predicted_labels))
        return total_eval_loss , task_auc

    def test(self):

        prediction_scores = np.array([])
        print(f"{'Predict task':*^80}")
        for batch in tqdm(self.test_loader, desc = "Predict Scores Loading", unit = "batch"):
            inputs = [data.to(self.device) for data in batch][0]

            with torch.no_grad():
                outputs = self.model(inputs)
                logits = F.softmax(outputs.to("cpu"), dim=1).data.numpy()
                y_pred = logits[:, 1]
                prediction_scores = np.append(prediction_scores, y_pred)

        self.save_predictions(prediction_scores)

    def save_predictions(self, predicts):
        file_path = "./finetuned_results/" + 'Model{}_BATCH{}_EPOCH{}_LR{}/'.format(self.model_name,
                  HYPERS['BATCH_SIZE'],HYPERS['EPOCHS'], HYPERS['LR'])
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        #输出当前时间
        dt = str(datetime.datetime.now()).split(" ")[0]
        filename = "{}_{}_(对应id和score).json".format(self.model_name, dt)
        with open(file_path + filename, "w") as fw:
            for idx, predict in enumerate(predicts):
                predict = {"id": idx, "label": predict}
                json.dump(predict, fw)
                if idx == len(predicts) - 1:
                    break
                fw.write("\n")

        filename2 = "{}_{}_result.tsv".format(self.model_name, dt)
        with open(file_path + filename2, "w") as fw:
            for idx, predict in enumerate(predicts):
                fw.write(str(predict))
                if idx == len(predicts) - 1:
                    break
                fw.write("\n")

    def save_stats(self ,stats):
        df = pd.DataFrame(stats)
        filename = 'Stats_{}_BATCH{}_Epoch{}_LR{}.csv'.format(self.model_name, HYPERS['BATCH_SIZE'],
                            HYPERS['EPOCHS'], HYPERS['LR'])
        df.to_csv("./finetuned_results/" + filename, sep = ",", encoding="utf-8", index = False)

    def save_model(self):
        # 保存训练好的模型
        model_name = 'Model{}_BATCH{}_Epoch{}_LR{}_TIME{}.pkl'.format(self.model_name, HYPERS['BATCH_SIZE'],
        HYPERS['EPOCHS'], HYPERS['LR'],time.strftime("%Y_%m_%d_%H_%M",time.localtime()))
        torch.save(self.model, "./finetuned_model/" + model_name)
        print("Fintuned model saved")


if __name__ == "__main__":
    parser

    ## 使用编号为4,5的两张显卡
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    # global hyper parameters
    HYPERS = {
        "BATCH_SIZE":32 * 4,
        "LR": 5e-5,
        "EPOCHS": 20,
    }
    seed_val = 2021
    corpus = Corpus(HYPERS["BATCH_SIZE"], seed_val)
    class_num = 2
    maxlen = corpus.max_len
    max_features = len(corpus.vocab)
    embedding_dims = 400
    dropout_rate = 0.2
    feature_size = 1024
    kernel_sizes = [2, 3, 4, 5]

    print("Build model.....")
    model = TEXTCNN(maxlen=maxlen,
                    max_features=max_features,
                    embedding_dims=embedding_dims,
                    class_num=class_num,
                    kernel_sizes=kernel_sizes,
                    dropout_rate=dropout_rate,
                    feature_size=feature_size,
                    vocab = corpus.v2i)
    if
    app = fintune(model, corpus)
    app.run_epoch()
    #app.save_model()