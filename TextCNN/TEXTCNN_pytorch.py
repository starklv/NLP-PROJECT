import torch
from torch import nn
from Spatial_Dropout import SpatialDropout
import gensim
import numpy as np




class TEXTCNN(nn.Module):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 kernel_sizes,
                 dropout_rate,
                 feature_size,
                 vocab):
        super().__init__()
        """
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param kernel_sizes: 滑动卷积窗口大小的list, eg: [1,2,3]
        :param class_num:二分类问题
        :param dropout_rate: 防止过拟合,使一部分权重不更新，抛弃一部分权重
        :param feature_size: 一位卷积层中输出通道的大小, 用于表示特征个数
        """
        self.maxlen = maxlen
        #self.embedding = nn.Embedding(num_embeddings=max_features, embedding_dim=embedding_dims)
        self.embedding_matrix = self.load_embedding(embedding_dims, vocab)
        ## 首先这里得对embedding的数据类型从torch.float64转化我torch.float32,若用torch.from_numpy就会转化为float64, double类型
        ## 还得设置词向量是否进行更新,
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.embedding_matrix, ))
        self.embedding.weight.requires_grad = True
        self.conv1ds = nn.ModuleList([nn.Conv1d(in_channels=embedding_dims, out_channels=feature_size, kernel_size = n,
                        stride=1,  padding=0)for n in kernel_sizes])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.avgpools = nn.ModuleList([nn.MaxPool1d(kernel_size = maxlen - n + 1) for n in kernel_sizes])
        self.dropout = SpatialDropout(dropout_rate)
        self.linear = nn.Linear(in_features = feature_size * len(kernel_sizes),
                                    out_features = 512)
        self.classifier = nn.Linear(in_features = 512,
                                    out_features = class_num)

    def forward(self, x):
        print(x.shape)
        if len(x.size()) != 2:
            raise ValueError("The rank of inputs of TextCNN must be 2, but now is {:d} ".format(len(x.size())))
        if x.size(1) != self.maxlen:
            raise ValueError(
                "The maxlen of inputs of TextCNN must be {0:d}, but now is {1:d}".format(self.maxlen, x.size(1)))

        emb_x = self.embedding(x)
        emb_x = self.dropout(emb_x)
        ##这一步很关键因为conv1d默认的输入通道在axis = 1的维度上,所以需要从
        #batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        emb_x = emb_x.permute(0, 2, 1)
        # 一维卷积后转变为 batch_size x out_channels x (text_len - kernel_size + 1)
        co = [self.relu(conv1d(emb_x)) for conv1d in self.conv1ds]
        # 一维池化之后转变为 batch_size x out_channels x 1
        co = [self.avgpools[i](co[i]) for i in range(len(co))]
        out = torch.cat(co, dim = 1)
        out = out.view(out.size(0), -1)
        out = self.relu(self.linear(out))
        out = self.classifier(out)
        return out

    def load_embedding(self, embedding_dims, word_vocab):
        Word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./Word2Vec/Word2Vec_400.bin', binary=True)
        embedding_matrix = np.zeros((len(word_vocab), embedding_dims))
        for word, i in word_vocab.items():
            embedding_vector = Word2vec_model.wv[word] if word in Word2vec_model else None
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                unk_vec = np.random.random(400) * 0.5
                unk_vec = unk_vec - unk_vec.mean()
                embedding_matrix[i] = unk_vec

        return embedding_matrix

