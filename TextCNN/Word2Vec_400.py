import pandas as pd
from gensim.models import Word2Vec
import os
import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds = elapsed_rounded))

train_data = pd.read_csv("../tianchi_datasets/track3_round1_newtrain3.tsv", sep="\t", header=None,
                                 names=["sentence1", "sentence2", "labels"])
train_data["document"] = train_data["sentence1"].str.cat(train_data["sentence2"], sep = " [SEP] ")
test_data = pd.read_csv("../tianchi_datasets/track3_round1_testA.tsv", sep="\t", header=None,
                                names=["sentence1", "sentence2"])
test_data["document"] = test_data["sentence1"].str.cat(test_data["sentence2"], sep=" [SEP] ")

print("train Word2Vec model .......")
t0 = time.time()
all_data = pd.concat([train_data["document"], test_data["document"]])
path = "./Word2Vec"
model_name = "Word2Vec_400.model"
## size生成词向量维度，window预测窗口大小，min_count总频率低于此值的单词忽略，sg = 1代表使用skip_gram
model = Word2Vec([[word for word in document.split(" ")] for document in all_data.values],
                size = 400, window = 5, iter = 10, workers=3, seed = 2021, min_count = 2, sg = 1)
##保存模型和词向量, 以二进制的格式保存
model.save(os.path.join(path, model_name)) #### 这样子保存的模型才可以继续训练
model.wv.save_word2vec_format(os.path.join(path, "Word2Vec_400.bin"), binary= True) ## 这样保存的只是词向量
train_time = format_time(time.time() - t0)
print(f"train time consume {train_time}")