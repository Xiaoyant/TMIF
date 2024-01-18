import copy
from torch import nn
from models import transformer


# Transformer encoder
class Transformer_encoder(nn.Module):
    def __init__(self, vocab=3000, sequence_length=296, N=4, d_model=128, d_ff=128, head=8, dropout=0.1):
        super(Transformer_encoder, self).__init__()
        self.vocab = vocab  # vocab：源数据特征(词汇)总数
        self.N = N  # N：编码器堆叠数
        self.d_model = d_model  # d_model：词向量映射维度
        self.d_ff = d_ff  # d_ff：前馈全连接网络中变化矩阵的维度
        self.head = head  # head：多头注意力机制中多头数
        self.dropout = dropout  # dropout：置零比率
        # 首先得到一个深度拷贝命令，接下来很多结构都要进行深度拷贝，从而保证它们彼此之间相互独立，不受干扰
        c = copy.deepcopy
        # 实例化位置编码器类
        self.position_encode = transformer.PositionalEncoding(d_model, dropout, sequence_length)
        # 实例化了多头注意力机制类，
        attn = transformer.MultiHeadAtten(head, d_model, dropout)
        # 然后实例化前馈全连接类
        ff = transformer.FeedForward(d_model, d_ff, dropout)
        # 实例化编码器
        self.encoderlayer = transformer.EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.encoder = transformer.Encoder(self.encoderlayer, N)

    def forward(self, x):
        # 位置编码
        input_x = self.position_encode(x)
        # encoder
        encoder_output = self.encoder(input_x)
        return encoder_output
