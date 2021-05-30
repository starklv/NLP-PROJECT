import pandas as pd
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


train_data = pd.read_csv("./tianchi_datasets/track3_round1_train.tsv", sep="\t", header=None,
                         quoting=3, encoding="utf-8", names=["sentence1", "sentence2", "labels"])
train_data["document"] = train_data["sentence1"].str.cat(train_data["sentence2"], sep=" ")
test_data = pd.read_csv("./tianchi_datasets/track3_round1_testA.tsv", sep="\t", header=None,
                        quoting=3, encoding="utf-8", names=["sentence1", "sentence2"])
test_data["document"] = test_data["sentence1"].str.cat(test_data["sentence2"], sep=" ")

docs = []
q_words = []
for word1, word2 in itertools.zip_longest(train_data["document"], test_data["document"]):
    docs.append(word1)
    if word2 is None:
        continue
    q_words.append(word2)

vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(docs)

tmp = []
for q in tqdm(q_words):
    qtf_idf = vectorizer.transform([q])
    res = cosine_similarity(tf_idf, qtf_idf).ravel()
    score = max(res)
    idx = np.argmax(res)
    if score > 0.95:
        label = train_data["labels"][idx]
    else:
        label = 2
    tmp.append(label)

test_data["labels"] = tmp
test_data1 = test_data[test_data["labels"] < 2]
new_data = pd.concat([train_data, test_data1], axis = 0)
new_data = new_data[["sentence1", "sentence2", "labels"]]
test_data1 = test_data1[["sentence1", "sentence2", "labels"]]
test_data1.to_csv("./tianchi_datasets/track3_round1_testA_label2.tsv", sep="\t", header=None, index=None)
new_data.to_csv("./tianchi_datasets/track3_round1_newtrain2.tsv", sep="\t", header=None, index=None)