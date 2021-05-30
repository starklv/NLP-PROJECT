import json
from bert4keras.tokenizers import load_vocab
from bert4keras.snippets import truncate_sequences


min_count = 5
maxlen = 100



def load_data(filename):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(' ')]
            b = [int(i) for i in b.split(' ')]
            truncate_sequences(maxlen, -1, a, b)
            D.append((a, b, c))
    return D

train_data = load_data(
    './tianchi_datasets/track3_round1_train.tsv'
)
test_data = load_data(
    './tianchi_datasets/track3_round1_testA.tsv'
)
dict_path = "./pretrain_model/chinese_roberta_wwm_ext/vocab.txt"


# 统计词频
tokens = {}
for d in train_data + test_data:
    for i in d[0] + d[1]:
        tokens[i] = tokens.get(i, 0) + 1

tokens = {i: j for i, j in tokens.items() if j >= min_count}
tokens = sorted(tokens.items(), key=lambda s: -s[1])

token_dict = load_vocab(dict_path)
with open('./counts.json', 'r') as f:
    counts = json.load(f)
del counts['[CLS]']
del counts['[SEP]']
#keep_tokens = {j:counts.get(i, 0) for i, j in sorted(token_dict.items(), key = lambda s:-s[1])}
keep_tokens = {i:counts.get(i, 0) for i, j in sorted(token_dict.items(), key = lambda s:-s[1])}
keep_tokens1 = sorted(keep_tokens.items(), key = lambda s:-s[1])
keep_tokens1 = keep_tokens[:len(tokens)]
#v2i = {tokens[i][0]:keep_tokens1[i][0] for i in range(len(tokens))}
v2i_chinese = {tokens[i][0]:keep_tokens1[i][0] for i in range(len(tokens))}

with open("v2i.json", "w") as fw:
    json.dump(v2i, fw)

with open('test.json', 'r') as f:
    data = json.load(f)