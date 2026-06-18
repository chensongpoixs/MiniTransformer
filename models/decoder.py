
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clones, SublayerConnection
from .layer_norm import LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        # 自注意力机制 ==> 注意self-attention的q，k和v均为decoder hidden
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        # 中间有3个子层：一个是多头自注意力机制，
        # 一个是与Encoder传入的Context进行Attention，
        # 另一个是前馈神经网络
        # ┌─────────────────────────────────────────────────────────┐
        # │                N × 解码器层                             │
        # │                (N × Decoder Layer)                      │
        # ├─────────────────────────────────────────────────────────┤
        # │  层 1:                                                  │
        # │  ┌────────────────────┐    ┌────────────────────┐     │
        # │  │ 多头自注意力       │ →  │ 与编码器上下文     │ →   │
        # │  │ (Multi-Head        │    │ 注意力机制         │
        # │  │  Self-Attention)   │    │ (Context Attention)│     │
        # │  └────────────────────┘    └────────────────────┘     │
        # │           ↑                          ↑                │
        # │           │ 残差连接                 │ 残差连接        │
        # │           │ + 层归一化               │ + 层归一化      │
        # │           │                          │                │
        # │      ┌────────────────────┐                     │
        # │      │ 前馈神经网络       │                     │
        # │      │ (Feed Forward)     │                     │
        # │      └────────────────────┘                     │
        # │                ↑ 残差连接                      │
        # │                │ + 层归一化                    │
        # └─────────────────────────────────────────────────────────┘
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory

        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
