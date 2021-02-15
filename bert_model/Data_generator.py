import os
from transformers import BertTokenizer
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
import tensorflow as tf


class data_generator:
    tokenizer = BertTokenizer.from_pretrained("E:/lv python/NLP/文本分类项目/bert-base-chinese/")

    def __init__(self, data_dir, batch_size):
        self.data_dir = "E:/lv python/NLP/天池热身赛_中文预训练语言模型/ocnli_public/"
        self.dic_data = {"entailment": 0, "neutral": 1, "contradiction": 2, "null": []}
        self.train = self.encode_examples(self.data_dir, "train.json").shuffle(10000).batch(batch_size)
        self.dev = self.encode_examples(self.data_dir, "dev.json").batch(batch_size)
        self.test = self.encode_examples(self.data_dir, "test.json").batch(batch_size)
        # self.tokenizer = BertTokenizer.from_pretrained("E:/lv python/NLP/文本分类项目/bert-base-chinese/")

    def convert_example_to_feature(self, review1, review2):
        return self.tokenizer.encode_plus(review1, review2, add_special_tokens=True,
                                          max_length=50, pad_to_max_length=True,
                                          return_attention_mask=True)

    def map_example_to_dict(self, input_ids, token_type_ids, attention_masks, label):
        return {"input_ids": input_ids,
                "token_types": token_type_ids,
                "attention_mask": attention_masks}, label

    def map_example_to_test(self, input_ids, token_type_ids, attention_masks):
        return {"input_ids": input_ids,
                "token_types": token_type_ids,
                "attention_mask": attention_masks}

    def encode_examples(self, data_dir, file):

        with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
            input_ids_list = []
            token_type_ids_list = []
            attention_mask_list = []
            label_list = []
            for line in tqdm(f):
                content = json.loads(line)
                label = content.get("label", "null")
                review1 = content["sentence1"].strip(".")
                review2 = content["sentence2"].strip(".")
                if label not in self.dic_data.keys():
                    continue
                bert_input = self.convert_example_to_feature(review1, review2)
                input_ids_list.append(bert_input["input_ids"])
                token_type_ids_list.append(bert_input["token_type_ids"])
                attention_mask_list.append(bert_input["attention_mask"])
                label_list.append(self.dic_data[label])
        if "test" in file:
            return tf.data.Dataset.from_tensor_slices((input_ids_list, token_type_ids_list,
                                                       attention_mask_list)).map(self.map_example_to_test)
        return tf.data.Dataset.from_tensor_slices((input_ids_list, token_type_ids_list,
                                                   attention_mask_list, label_list)).map(self.map_example_to_dict)