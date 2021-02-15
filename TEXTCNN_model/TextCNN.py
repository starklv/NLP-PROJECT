from tensorflow import keras


class TextCNN(keras.Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 kernel_sizes=[1, 2, 3],
                 last_activation="softmax"):
        '''
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param kernel_sizes: 滑动卷积窗口大小的list, eg: [1,2,3]
        :param class_num:二分类问题
        '''
        super().__init__()
        self.maxlen = maxlen
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.embedding = keras.layers.Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.conv1ds = [keras.layers.Conv1D(filters=128, kernel_size=n, activation="relu") for n in kernel_sizes]
        self.avgpools = [keras.layers.GlobalMaxPooling1D() for _ in range(len(kernel_sizes))]
        self.classifier = keras.layers.Dense(units=class_num, activation=last_activation)

    def call(self, inputs):
        if len(inputs.shape) != 2:
            raise ValueError("The rank of inputs of TextCNN must be 2, but now is {:d} ".format(len(inputs.shape)))
        if inputs.shape[1] != self.maxlen:
            raise ValueError(
                "The maxlen of inputs of TextCNN must be {0:d}, but now is {1:d}".format(self.maxlen, inputs.shape[1]))

        emb = self.embedding(inputs)
        co = [conv1d(emb) for conv1d in self.conv1ds]
        co = [self.avgpools[i](co[i]) for i in range(len(co))]
        x = keras.layers.Concatenate()(co)
        outputs = self.classifier(x)
        return outputs



