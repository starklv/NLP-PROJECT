import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from TextCNN import TextCNN

'''
:param maxlen: 文本最大长度
:param max_features: 词典大小
:param embedding_dims: embedding维度大小
:param kernel_sizes: 滑动卷积窗口大小的list, eg: [1,2,3]
:param class_num:二分类问题
'''
class_num = 2
maxlen = 400
max_features = 5000
embedding_dims = 100
epochs = 5
batch_size = 6

# IMDB数据集包含来自互联网的50000条严重两极分化的评论，该数据被分为用于训练的25000条评论和用于测试的25000条评论，
# 训练集和测试集都包含50%的正面评价和50%的负面评价。该数据集已经经过预处理：评论（单词序列）已经被转换为整数序列，
# 其中每个整数代表字典中的某个单词。
# 参数num_words = 10000的意思是仅保留训练数据的前10000个最常见出现的单词，低频单词将被舍弃。这样得到的向量数据不会太大，便于处理。
print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(path = "C:/Users/Administrator/.keras/datasets/imdb.npz", num_words = max_features)

print("train sequences: {:d}".format(len(x_train)))
print("test sequences : {:d}".format(len(x_test)))

# pad_sequences 就是把不等长的list变成等长, 默认填充是0，padding = "post"是前填充，padding = "pre"是后填充
x_train = pad_sequences(x_train, maxlen=maxlen, padding="post")
x_test = pad_sequences(x_test, maxlen=maxlen, padding="post")
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# from_tensor_slices切分传入的numpy，将其转化为dataset,并且一对一匹配词向量x和label y,成为一组，同时方便后续使用batch函数(dataset数据类型)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

print("Build model.....")
model = TextCNN(maxlen=maxlen,
                max_features=max_features,
                embedding_dims=embedding_dims,
                class_num=class_num,
                kernel_sizes=[2, 3, 5],
                last_activation="softmax")

# 为训练器选择优化器和损失函数
loss_object = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(0.01)

# 选择衡量指标来度量模型的损失至(loss)和准确率(accuracy)。这些指标在epoch上累积值,然后打印出整体结果
train_loss = keras.metrics.Mean()
train_accuracy = keras.metrics.SparseCategoricalAccuracy()
test_loss = keras.metrics.Mean()
test_accuracy = keras.metrics.SparseCategoricalAccuracy()


# 使用tf.GradientTape来训练模型
@tf.function  # 将即时执行模型转化为图执行模式，加快训练速度
def train_step(x, labels):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# 测试模型
@tf.function
def test_step(x, labels):
    predictions = model(x)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


def inference(x):
    predictions = model(x)
    return np.argmax(predictions, axis=1)


print("Train.....")

for epoch in range(epochs):
    # 在下一次epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for x, labels in tqdm(train_ds):
        train_step(x, labels)

    for x, labels in tqdm(test_ds):
        test_step(x, labels)

    template = "Epoch {:d}, Train Loss: {:.2f}, Train Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}"
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

print("Test....")
pred = np.array([])
true = np.array([])
for x, test_labels in tqdm(test_ds):
    pred = np.append(pred, inference(x))
    true = np.append(true, test_labels)

##这部分是词汇表，将下标索引idx转化为电影影评
word_index = imdb.get_word_index(path = "E:/lv python/NLP/文本分类项目/keras内置datasets/imdb_word_index.json")
i2v = {i:v for v, i in word_index.items()}
print(" ".join([i2v.get(i, "?") for i in train_x[0].numpy()]))