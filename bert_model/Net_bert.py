from tensorflow import keras
from transformers import TFBertModel
import tensorflow as tf



class bert_modified(keras.Model):
    def __init__(self, class_num, last_activation, pretrained_model):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(pretrained_model)
        self.task = keras.layers.Dense(units=class_num, activation=last_activation)

    def call(self, inputs):
        if len(inputs["input_ids"].shape) != 2:
            raise ValueError("The rank of inputs of bert must be 2, but now is {:d}").format(len(inputs["inputs_ids"].shape))
        if inputs["input_ids"].shape[1] != 50:
            raise ValueError(
                "The maxlen of inputs of TextCNN must be {0:d}, but now is {1:d}".format(inputs["input_ids"].shape[1]))
        z = self.bert(inputs).last_hidden_state
        # z = self.bert(inputs).pooler_output
        # 用上面这种方式输出的话, z代表的是第一个分类标记的最后一层隐藏层的输出,涵盖的信息不全;
        # 但用了下面这种表达方式的话，表示的是整个序列的最后一层隐藏层的输出，涵盖了所有的语义信息,同时得转换成二维形状，方便接全连接层
        outs = self.task(tf.reshape(z, [z.shape[0], -1]))
        return outs