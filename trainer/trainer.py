

import torch
import torch.nn as nn
import numpy as np
import math
import copy




from models.transformer import Transformer
from models.encoder import Encoder, EncoderLayer
from models.decoder import Decoder, DecoderLayer
from models.generator import Generator
from models.positional_encoding import PositionalEncoding
from models.attention import MultiHeadedAttention
from models.feed_forward import PositionwiseFeedForward
from models.embedding import Embeddings


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
构建完整的Transformer模型
'''
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    # 使用深拷贝函数
    # 只是将函数对象赋值给变量，而不是执行拷贝操作。要真正拷贝数据，需要调用这个函数：
    # c = copy.deepcopy  # 只是引用
    # copied_data = c(original_data)  # 这才是真正的拷贝
    c = copy.deepcopy;
    #   tokenerizer  词语嵌入  位置编码
    # 实例化Attention对象  多注意力机制实现  4全连接 函数
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)

    # 实例化FeedForward对象
    # 前向反馈神经网络实现
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象 偶数维度使用sin函数，奇数维度使用cos函数
    position = PositionalEncoding(d_model, dropout).to(DEVICE)

    #
    #




    # 实例化Transformer模型对象
    '''
     Transformer模型的整体结构
    '''
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE), # EncoderLayer ==> encoder layer
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),  #  DecoderLayer ==> decoder layer
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),  # src_embed ==> source embedding
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),  # tgt_embed ==> target embedding
        Generator(d_model, tgt_vocab)).to(DEVICE)  # generator ==> 生成器

    # 初始化模型参数
    # 遍历模型中的所有参数
    for p in model.parameters():
        # 判断参数是否为二维或更高维（例如权重矩阵，而不是偏置向量）
        # print(f"name: {p.name}, requires_grad: {p.requires_grad}")
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    print("===========================================================================")
    for name, p in model.named_parameters():
        print(f"name: {name}, shape: {p.shape}, requires_grad: {p.requires_grad}")
    print("===========================================================================")
    return model.to(DEVICE)
