import json
import pandas as pd
import os

test_score = "tianchi_datasets/test.json"
train_data = "tianchi_datasets/track3_round1_train.tsv"
test_data = "tianchi_datasets/track3_round1_testA.tsv"
def create_new_traindata(test_score, train_data, test_data):
    tmp = []
    dir_path = os.getcwd()

    with open(os.path.join(dir_path, test_score), "r", encoding="utf-8") as f:
        for line in f:
            content = json.loads(line)
            score = content["label"]
            idx = content["id"]
            if score > 0.95:
                tmp.append([idx, 1])
            elif score < 0.05:
                tmp.append([idx, 0])
            else:
                tmp.append([idx, "unk"])

    with open(os.path.join(dir_path, test_data), "r", encoding="utf-8") as fr:
        with open("tianchi_datasets/track3_round1_testA_label.tsv", "w", encoding="utf-8") as fw:
            data = fr.readlines()
            for i in range(len(data)):
                content = data[i].strip()
                label = tmp[i][1]
                if label in [0, 1]:
                    lis = content + "\t" + str(label) + "\n"
                else:
                    lis = ""
                fw.write(lis)

    train = pd.read_csv(os.path.join(dir_path, train_data), sep="\t", header=None,
                             names=["sentence1", "sentence2", "labels"])
    test = pd.read_csv("./tianchi_datasets/track3_round1_testA_label.tsv", sep="\t", header=None,
                            names=["sentence1", "sentence2", "labels"])

    new_data = pd.concat([train, test], axis=0)
    new_data.to_csv("./tianchi_datasets/track3_round1_newtrain.tsv", sep="\t", header=None, index=None)


""" 查看训练集和测试集的分布是否一致
train["sentence1"].str.len().describe(percentiles = [.99, .999, .9999]), train["sentence2"].str.len().describe(percentiles = [.99, .999, .9999])\
,test["sentence1"].str.len().describe(percentiles = [.99, .999, .9999]), test["sentence2"].str.len().describe(percentiles = [.99, .999, .9999])
"""

