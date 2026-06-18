

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import  copy
import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# def clones(module, N):
#     """克隆模型块，克隆的模型块参数不共享"""
#     # copy.deepcopy 用于创建模块的N个独立副本 ==>
#     # 这些副本在训练过程中将拥有各自独立的参数，不会相互影响
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
from  .utils import clones


"""
这个函数是Transformer模型的基础，它实现了模型的核心注意力机制
"""
def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    """
        Encoder：主要使用padding mask处理不等长序列
        Decoder：同时使用padding mask和sequence mask
        padding mask处理填充部分
        sequence mask防止看到未来信息
    """
    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)

    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


"""
这个MultiHeadedAttention类,实现多头注意力机制
"""
class MultiHeadedAttention(nn.Module):
    """

    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__();
        # 保证可以整除
        assert d_model % h == 0;
        # 得到一个head的attention表示维度
        self.d_k = d_model // h;
        # d_model打印信息 512  h打印信息 8  d_k打印信息 64
        print("MultiHeadedAttention: d_model={}, h={}, d_k={}".format(d_model, h, self.d_k));
        # head数量
        self.h = h;
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4);
        self.attn = None;
        #  dropout层 用于防止过拟合
        self.dropout = nn.Dropout(p=dropout);

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1);
        # query的第一个维度值为batch size
        nbatches = query.size(0);
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))];
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout);
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k);
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x);
