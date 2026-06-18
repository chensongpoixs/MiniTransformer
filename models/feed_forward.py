
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Transformer模型中前馈神经网络的实现。
"""
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        位置前馈神经网络初始化函数
        参数:
            d_model: 模型的输入维度
            d_ff: 前馈神经网络中间层的维度
            dropout: dropout概率，默认为0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        # Linear层，将输入维度从d_model映射到d_ff
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一个线性层，将维度从d_model扩展到d_ff
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二个线性层，将维度从d_ff压缩回d_model
        self.dropout = nn.Dropout(dropout)    # dropout层，用于防止过拟合

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # 先通过第一个线性层，然后应用ReLU激活函数，
