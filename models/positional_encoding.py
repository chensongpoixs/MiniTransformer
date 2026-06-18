
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
为每个输入位置添加一个唯一的位置编码
这个编码会被添加到词向量中，使模型能够理解位置信息
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 防止过离合
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的positional embedding
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        print("pe shape:", pe.shape)  # pe shape: torch.Size([5000, 512])
        print("pe:", pe);
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        # torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        print("PositionalEncoding: d_model={}, max_len={}".format(d_model, max_len));
        print("position shape:", position.shape)  # position shape: torch.Size([5000, 1])
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))

        # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        print("pe after even:", pe);
        pe[:, 1::2] = torch.cos(position * div_term)
        print("pe after odd:", pe);
        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


