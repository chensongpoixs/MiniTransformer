


import torch
import torch.nn as nn






"""
定义一个层归一化的类
"""
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 这是一个前向传播函数，用于执行Layer Normalization操作
        # 输入x是神经网络层的输出
        # 按最后一个维度计算均值和方差
        # keepdim=True确保输出的维度与输入相同
        mean = x.mean(-1, keepdim=True)  # 计算最后一个维度的均值
        std = x.std(-1, keepdim=True)   # 计算最后一个维度的标准差

        # 返回Layer Norm的结果
        # Layer Norm公式: y = a * (x - mean) / sqrt(std^2 + eps) + b
        # 其中a和b是可学习的参数，eps是为了防止除以0的小常数
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2
