


import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


from  .utils import clones, SublayerConnection
from .layer_norm import LayerNorm



class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # 多注意力机制
        self.self_attn = self_attn
        # 前馈神经网络
        self.feed_forward = feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        #  中间有2个子层：一个是多头自注意力机制，另一个是前馈神经网络
        '''
        ┌─────────────────────────────────────────────────────────┐
        │                N × 编码器层                             │
        │                (N × Encoder Layer)                      │
        ├─────────────────────────────────────────────────────────┤
        │  层 1:                                                  │
        │  ┌────────────────────┐    ┌────────────────────┐     │
        │  │ 多头自注意力          │ →  │ 前馈神经网络         │     │
        │  │ (Multi-Head        │    │ (Feed Forward)     │     │
        │  │  Self-Attention)   │    └────────────────────┘     │
        │  └────────────────────┘              ↑                │
        │           ↑                          │ 残差连接        │
        │           │ 残差连接                 │ + 层归一化      │
        │           │ + 层归一化               │                │
        └─────────────────────────────────────────────────────────┘
        '''
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)




class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
