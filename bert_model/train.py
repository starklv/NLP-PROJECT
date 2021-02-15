from Net_bert import bert_modified
from Data_generator import data_generator
from tensorflow import keras
import numpy as np
import tensorflow as tf
import time


import logging
logging.disable(30)
"""可避免出现此类报错,但bert最后一层的梯度穿不过去的问题该如何解决还是未知
WARNING:tensorflow:Gradients do not exist for variables ['tf_bert_model/bert/pooler/dense/kernel:0', 
'tf_bert_model/bert/pooler/dense/bias:0'] when minimizing the loss.
"""


def train_step(x, labels, loss_object, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


def test_step(x, labels, loss_object):
    predictions = model(x)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


def inference(x):
    predictions = model(x)
    return np.argmax(predictions, axis=1)


def train(data, epochs):
    train_ds = data.train
    val_ds = data.dev
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(0.01)
    t0 = time.time()
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for x, labels in train_ds:
            train_step(x, labels, loss_object, optimizer)

        for x, labels in val_ds:
            test_step(x, labels, loss_object)

        t1 = time.time()
        print("Epoch: {:d}".format(epoch + 1),
              "| time_per_epoch: {:.2f}".format(t1 - t0),
              "| Train Loss: {:.2f}".format(train_loss.result()),
              "| Train Accuracy: {:.2f}".format(train_accuracy.result() * 100),
              "| Test Loss: {:.2f}".format(test_loss.result()),
              "| Test Accuracy: {:.2f}".format(test_accuracy.result() * 100),
              )
        t0 = t1



if __name__ == "__main__":
    class_num = 3
    last_activation = "softmax"
    pretrained_model = "E:/lv python/NLP/文本分类项目/bert-base-chinese/"
    data_dir = "E:/lv python/NLP/天池热身赛_中文预训练语言模型/ocnli_public/"
    max_len = 50
    batch_size = 4
    epochs = 10
    print(f"''")
    d = data_generator(data_dir, batch_size)
    model = bert_modified(class_num, last_activation, pretrained_model)
    train_loss = keras.metrics.Mean()
    train_accuracy = keras.metrics.SparseCategoricalAccuracy()
    test_loss = keras.metrics.Mean()
    test_accuracy = keras.metrics.SparseCategoricalAccuracy()
    print(f"{'Train':*^80}")
    train(d, epochs = epochs)

#因为pred是arrray形式或者是[]形式,直接用list的话,每次子元素是列表,[[1, 1, 1, 0]]这种形式,
#但如果用array形式的话,相当于自动把子元素中的元素添加进去了,array([1, 1, 1, 0])
"""
pred = np.array([])
true = np.array([])
for x, labels in d.dev:
    pred = np.append(pred, inference(x))
    true = np.append(true, labels)
"""