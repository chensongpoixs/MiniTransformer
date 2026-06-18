

import  torch
import torch.nn as nn
import math


"""
将输入的离散词索引转换为连续的向量表示
例如，将词汇表中的第5个词映射为一个512维的向量
"""
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # 初始化方法，传入模型的维度（d_model）和词汇表的大小（vocab）
        super(Embeddings, self).__init__()
        # Embedding层，将词汇表的大小映射为d_model维的向量
        # 词表， 和维度  生成词项矩阵
        self.lut = nn.Embedding(vocab, d_model)
        # 存储模型的维度 d_model
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        # 这是为了保持词向量的方差，使其适应后续层的训练。
        return self.lut(x) * math.sqrt(self.d_model)