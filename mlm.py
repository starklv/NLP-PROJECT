import pandas as pd
import numpy as np
import json

MASK_RATE = 0.15
with open('./v2i.json', 'r') as f:
    v2i = json.load(f)


def _get_loss_mask(len_arange, seq, pad_id):
    rand_id = np.random.choice(len_arange, size=max(2, int(MASK_RATE * len(len_arange))), replace=False)
    loss_mask = np.full_like(seq, pad_id, dtype=np.bool)
    loss_mask[rand_id] = True
    return loss_mask[None, :], rand_id

def do_mask(seq, len_arange, pad_id, mask_id):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = mask_id
    return loss_mask

def do_nothing(seq, len_arange, pad_id):
    loss_mask, _ = _get_loss_mask(len_arange, seq, pad_id)
    return loss_mask

def do_replace(seq, len_arange, pad_id, word_ids):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = np.random.choice(word_ids, size=len(rand_id))
    return loss_mask


train_data = pd.read_csv("./tianchi_datasets/track3_round1_newtrain3.tsv", sep="\t", header=None,
                                 names=["sentence1", "sentence2", "labels"])
train_data["document"] = train_data["sentence1"].str.cat(train_data["sentence2"], sep = " [SEP] ")
train_data["document"] = train_data["document"].map(lambda x:[word for word in x.split(" ")])


input_ids = []
max_length = 93
for inputs in train_data["document"]:
    arange = np.arange(0, len(inputs) + 2)
    padded = np.full(max_length, fill_value=0, dtype=np.int32)
    padded[:(len(inputs) + 2)] = [v2i['[CLS]']] + [v2i.get(x, v2i['[UNK]']) for x in inputs] + [v2i['[SEP]']]
    p = np.random.random
    if p < 0.7:
        loss_mask = do_mask(padded, arange, v2i["[PAD]"], v2i["[MASK]"])

    break