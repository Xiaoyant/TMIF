import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math


# 2.位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, sequence_length):
        super(PositionalEncoding, self).__init__()
        # 实例化dropout层
        self.dpot = nn.Dropout(p=dropout)
        # 初始化位置编码矩阵
        pe = torch.zeros(sequence_length, embedding_size)
        # 初始化绝对位置矩阵
        # position矩阵size为(max_len,1)
        position = torch.arange(0, sequence_length).unsqueeze(1)
        # 将绝对位置矩阵和位置编码矩阵特征融合
        # 定义一个变换矩阵 跳跃式初始化
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * -(math.log(10000) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将二维张量扩充成三维张量
        pe = pe.unsqueeze(0)
        # 把pe位置编码矩阵注册成模型的buffer
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要说着优化步骤进行更新的增益
        # 注册之后我们就可以在模型保存后重新加载时盒模型结构与参数已通被加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        text_input = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dpot(text_input)


# 3.注意力机制
def subsequent_mask(size):
    atten_shape = (1, size, size)
    # 对角线下就是负 对角线上就是正 对角线就是0
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    return torch.from_numpy(1 - mask)


def attention(query, key, value, mask=None, dropout=None):
    """:param query:
    :param key:
    :param value:
    :param mask:  掩码张量
    :param dropout:
    :return: query在key和value作用下的表示"""
    # 获取query的最后一维的大小，一般情况下就等同于我们的词嵌入维度
    d_k = query.size(-1)
    # 按照注意力公式，将query与key转置相乘，这里面key是将最后两个维度进行转置，再除以缩放系数
    # 得到注意力的得分张量score
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 使用tensor的masked_fill方法，将掩码张量和scores张量每个位置一一比较
        # 如果掩码张量处为0 则对应的score张量用-1e9替换
        score = score.masked_fill(mask == 0, -1e9)

    p_atten = F.softmax(score, dim=-1)

    if dropout is not None:
        p_atten = dropout(p_atten)
    # 返回注意力表示
    return torch.matmul(p_atten, value.float()), p_atten


# 4.多头注意力机制
def Clone(module, N):
    """
    生成相同的网络层的克隆函数
    :param module:  目标网络层
    :param N: 克隆数量
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAtten(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        :param head: 头数
        :param embedding_dim: 词嵌入维度
        :param dropout:
        """
        super(MultiHeadAtten, self).__init__()

        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被此嵌入维度整除
        # 因为要给每个头分配等量的词特征
        assert embedding_dim % head == 0
        # 得到每个头获得的分割词向量维度d_K
        self.d_k = embedding_dim // head
        # 获得头数
        self.head = head
        # 克隆四个全连接层对象，通过nn的Linear实例化
        self.linears = Clone(nn.Linear(embedding_dim, embedding_dim), 4)
        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以未None
        self.attn = None
        # 最后就是一个self.dropout对象
        self.dpot = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 前向逻辑函数
        if mask is not None:
            # 扩展维度 代表对头中的第i个头
            mask = mask.unsqueeze(1)
        # 获取样本数
        batch_size = query.size(0)

        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV与三个全连接层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头风格输入，使用view方法对线性变换的结果进行维度重塑，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度
        # 计算机会根据这种变换自动计算这里的值，然后对第二维和第三维进行转置操作
        # lis = [query,key,value]
        # r = []
        # for step,model in enumerate(self.linears):
        #     r.append(model(lis[step]))
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for model, x in
                             zip(self.linears, (query, key, value))]
        # 得到每个头的输入后，接下来就是将他们传入到attention中,
        # 这里直接attention函数
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dpot)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的四维张量，我们需要将其转换为输入的形式
        # 对 第2，3维进行转置 然后使用contiguous方法
        # 这个方法的作用就是能够让转置后的张量应用view方法
        # contiguous()这个函数，把tensor变成在内存中连续分布的形式
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 使用线性层列表中的最后一个线性层对输入进行线性变
        return self.linears[-1](x)


# 5.前馈全连接层
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff, dropout=0.1):
        """
        :param d_model: 线性层输入维度
        :param d_ff: 线性层输出维度
        :param dropout:
        """
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(embedding_dim, d_ff)
        self.w2 = nn.Linear(d_ff, embedding_dim)
        self.dpot = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dpot(F.relu(self.w1(x))))


# 6.规范化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        :param features:  代表词嵌入的维度
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        # 防止分母为0
        self.eps = eps

    def forward(self, x):
        # 对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致
        mean = x.mean(-1, keepdim=True)
        # 接着再求最后一个维度的标准差
        std = x.std(-1, keepdim=True)
        # 然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果
        # 最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进行乘法操作，加上位移参数
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


# 7.子层连接结构
class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # size:  词嵌入维度的大小
        super(SubLayerConnection, self).__init__()
        # 实例化规范化对象
        self.norm = LayerNorm(size)
        self.dpot = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        接受上一个层或者子层的输入作为第一个参数
        将该子层连接中的子层函数作为第二个参数
        :param x:
        :param sublayer:
        :return:
        """
        # 首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作
        # 随机停止一些网络中神经元的作用 防止过拟合
        # 残差连接
        return x + self.dpot(sublayer(self.norm(x)))


# 8.编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        # size：词嵌入维度的大小
        # self_attn：传入多头注意力子层实例化对象，并且是自注意力机制
        # feed_forward：前馈全连接层实例化对象
        # dropout：置零比率
        super(EncoderLayer, self).__init__()
        self.size = size
        # 首先将self_attn和feed_forward传入其中
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        # 编码器层中有两个子层连接结构，所以使用clones函数进行克隆
        self.sublayer = Clone(SubLayerConnection(size, dropout), 2)

    def forward(self, x):
        # x：上一层输出
        # mask：掩码张量
        # 里面就是按照结构图左侧的流程，首先通过第一个子层连接结构，其中包含多头注意力子层，然后通过第二个子层连接结构，
        # 其中包含前馈全连接子层，最后返回结果
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


# 9.编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        # layer：编码器层
        # N: 编码器层的个数
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = Clone(layer, N)
        # 再初始化一个规范化层，将用在编码器的最后面
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        # forward函数的输入和编码器层相同，x代表上一层的输出，mask代表掩码张量
        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x
        # 这个循环的过程，就相当于输出的x经过N个编码器层的处理
        # 最后在通过规范化层的对象self.norm进行处理，最后返回结果
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# 10.解码器层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0.1):
        # size：词嵌入的维度大小，同时也代表解码器层的尺寸
        # self_attn：多头注意力对象，也就是说这个注意力机制需要Q=K=V
        # src_attn：多头注意力对象，这里Q!=K=V
        # feed_forward：前馈全连接层对象
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        # 按照结构图使用clones函数克隆三个子层连接对象
        self.sublayer = Clone(SubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        # x：来自上一层的输出
        # mermory：来自编码器层的语义存储变量
        # 源数据掩码张量和目标数据掩码张量
        m = memory
        # 将x传入第一个子层结构，第一个子层结构分别是x和self-attn函数，因为是自注意力机制，所以Q,K,V都是x
        # 最后一个参数的目标数据掩码张量，这是要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据
        # 比如扎起解码器准备生成第一个字符或者词汇时，我们其实已经传入了第一个字符以便计算损失
        # 但是我们不希望在生成第一个字符时模型还能利用这个信息，因此我们会将其遮掩，同样生成第二个字符或词汇时
        # 模型只能使用第一个字符或者词汇信息，第二个字符以及之后的信息都不允许模型使用
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 接着进入第二个子层，这个职称中常规的注意力机制，q是输入x；k,v是编码层输出的memory
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄露，而是这笔掉对结果没有意义的字符而陈胜的注意力值
        # 以此提升模型效果和训练速度，这样就完成了第二个子层的处理
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        # 最后一个子层就是前馈全连接子层，经过它的处理后，就可以返回结果，这就是我们的解码器层结构。
        return self.sublayer[2](x, self.feed_forward)


# 11.解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        # layer：解码器层layer
        # N：解码器层的个数N
        super(Decoder, self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化了一个规范化层
        # 因为数据走过了所有的解码器层后，最后要做规范化处理
        self.N = N
        self.layers = Clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        # x：数据的嵌入表示
        # memory：编码器层的输出
        # source_mask：源数据掩码张量
        # target_mask：目标数据掩码张量

        # 对每个层进行循环，淡然这个循环就是变量x通过每一个层的处理
        # 得出最后的结果，再进行一次规范化返回即可
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# 12.输出层
class Generator(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        # d_model：词嵌入维度
        # vocab_size：词表大小
        super(Generator, self).__init__()
        # 首先就是使用nn中的预定义线形层进行实例化，得到一个对象self.project等待使用
        # 这个线性层的参数有两个，计时初始化函数时传进来的 d_model和vocab_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 前向逻辑函数中输入是上一层输出张量x
        # 在函数中，首先使用上一步得到的self.project对x进行线性变化
        # 然后使用F中已经实现的log_softmax进行的softmax处理
        return F.log_softmax(self.project(x), dim=-1)


if __name__ == '__main__':
    vocab = 3000
    d_model = 128
    dropout = 0.2
    inputs = torch.randint(low=0, high=100, size=(8, 100), dtype=torch.long)
    # 实例化文本嵌入层对象
    TE = Embedding(vocab, d_model)
    # 实例化位置编码层
    PE = PositionalEncoding(d_model, dropout, sequence_length=100)
    TER = TE(inputs)
    print(TER.shape)
    PER = PE(TER)
    print(PER.shape)
    # 实例化多头注意力机制
    head = 8
    MHA = MultiHeadAtten(head=head, embedding_dim=d_model, dropout=dropout)
    # MHAR = MHA(PER,PER,PER)
    # print(MHAR.shape)
    # 实例化规范化层
    # LN = LayerNorm(d_model)
    # LNR = LN(MHAR)
    # print(LNR.shape)
    MHAR = lambda x: MHA(x, x, x)
    # 实例化 子层连接结构
    SLC1 = SubLayerConnection(d_model, dropout=dropout)
    SLC1R = SLC1(PER, MHAR)
    print(SLC1R.shape)
    # 实例化 前馈全连接层
    FF = FeedForward(d_model, 1024, dropout=dropout)
    SLC2 = SubLayerConnection(d_model, dropout=dropout)
    SLC2R = SLC2(SLC1R, FF)
    print(SLC2R.shape)
