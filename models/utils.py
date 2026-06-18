
import torch
import torch.nn as nn
import  copy
# 方式一：直接导入类（推荐）
from .layer_norm import LayerNorm
# 方式二：导入包，通过包访问类
# import layer_norm
# norm = layer_norm.LayerNorm(features=512)


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    # copy.deepcopy 用于创建模块的N个独立副本 ==>
    # 这些副本在训练过程中将拥有各自独立的参数，不会相互影响
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])







"""
SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
"""
class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层
    连在一起只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    """
    sublayer参数是SublayerConnection类中的核心组件，
    它代表了Transformer中的具体处理层，通过这种设计实现了代码的模块化和灵活性。
    """
    def forward(self, x, sublayer):
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))