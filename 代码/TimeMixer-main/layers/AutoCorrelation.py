import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络的功能模块
import matplotlib.pyplot as plt  # 导入 Matplotlib 用于绘图
import numpy as np  # 导入 NumPy 库用于数值计算
import math  # 导入数学库
from math import sqrt  # 从数学库中导入平方根函数
import os  # 导入操作系统相关模块


class AutoCorrelation(nn.Module):  # 定义自相关机制类，继承自 nn.Module
    """
    自相关机制类，包含以下两个阶段：
    (1) 基于周期的依赖关系发现
    (2) 时间延迟聚合
    此模块可以无缝替换自注意力机制家族。
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        # 初始化函数，设置类的参数和超参数
        super(AutoCorrelation, self).__init__()  # 调用父类的初始化方法
        self.factor = factor  # 因子，用于调整自相关计算
        self.scale = scale  # 缩放因子
        self.mask_flag = mask_flag  # 是否使用掩码的标志
        self.output_attention = output_attention  # 是否输出注意力权重的标志
        self.dropout = nn.Dropout(attention_dropout)  # 定义 Dropout 层

    def time_delay_agg_training(self, values, corr):  # 定义时间延迟聚合的训练版本
        """
        自相关的加速版本（类似批归一化的设计）
        此函数用于训练阶段。
        """
        head = values.shape[1]  # 获取 head 的数量
        channel = values.shape[2]  # 获取通道数量
        length = values.shape[3]  # 获取序列长度
        # 找到前 k 个最大的值
        top_k = int(self.factor * math.log(length))  # 计算 top k 值
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # 计算相关性的平均值
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]  # 找到相关性最高的位置索引
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # 获取对应权重
        # 更新相关性
        tmp_corr = torch.softmax(weights, dim=-1)  # 对权重进行 softmax 操作
        # 聚合过程
        tmp_values = values  # 暂存原始值
        delays_agg = torch.zeros_like(values).float()  # 初始化延迟聚合结果
        for i in range(top_k):  # 遍历 top k 值进行聚合
            pattern = torch.roll(tmp_values, -int(index[i]), -1)  # 按照延迟索引滚动（移动）数值
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
            # 更新聚合结果
        return delays_agg  # 返回延迟聚合结果

    def time_delay_agg_inference(self, values, corr):  # 定义时间延迟聚合的推断版本
        """
        自相关的加速版本（类似批归一化的设计）
        此函数用于推断阶段。
        """
        batch = values.shape[0]  # 获取批量大小
        head = values.shape[1]  # 获取 head 的数量
        channel = values.shape[2]  # 获取通道数量
        length = values.shape[3]  # 获取序列长度
        # 索引初始化
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # 找到前 k 个最大的值
        top_k = int(self.factor * math.log(length))  # 计算 top k 值
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # 计算相关性的平均值
        weights, delay = torch.topk(mean_value, top_k, dim=-1)  # 找到相关性最高的延迟值和权重
        # 更新相关性
        tmp_corr = torch.softmax(weights, dim=-1)  # 对权重进行 softmax 操作
        # 聚合过程
        tmp_values = values.repeat(1, 1, 1, 2)  # 扩展数值向量
        delays_agg = torch.zeros_like(values).float()  # 初始化延迟聚合结果
        for i in range(top_k):  # 遍历 top k 值进行聚合
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            # 计算延迟索引
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)  # 根据索引提取模式
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
            # 更新聚合结果
        return delays_agg  # 返回延迟聚合结果

    def time_delay_agg_full(self, values, corr):  # 定义标准版时间延迟聚合函数
        """
        标准版自相关
        """
        batch = values.shape[0]  # 获取批量大小
        head = values.shape[1]  # 获取 head 的数量
        channel = values.shape[2]  # 获取通道数量
        length = values.shape[3]  # 获取序列长度
        
        # 索引初始化
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        
        # 找到前 k 个最大的值
        top_k = int(self.factor * math.log(length))  # 计算 top k 值
        weights, delay = torch.topk(corr, top_k, dim=-1)  # 找到相关性最高的延迟值和权重
        
        # 更新相关性
        tmp_corr = torch.softmax(weights, dim=-1)  # 对权重进行 softmax 操作
        
        # 聚合过程
        tmp_values = values.repeat(1, 1, 1, 2)  # 扩展数值向量
        delays_agg = torch.zeros_like(values).float()  # 初始化延迟聚合结果
        
        for i in range(top_k):  # 遍历 top k 值进行聚合
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)  # 计算延迟索引
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)  # 根据索引提取模式
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))  # 更新聚合结果
        
        return delays_agg  # 返回延迟聚合结果

    def forward(self, queries, keys, values, attn_mask):  # 定义前向传播函数
        B, L, H, E = queries.shape  # 获取查询的形状 (批量, 序列长度, head 数量, 嵌入维度)
        _, S, _, D = values.shape  # 获取数值的形状 (批量, 序列长度, head 数量, 通道数)
        
        if L > S:  # 如果查询序列长度大于数值序列长度
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()  # 创建零填充
            values = torch.cat([values, zeros], dim=1)  # 对数值进行零填充
            keys = torch.cat([keys, zeros], dim=1)  # 对键进行零填充
        else:  # 否则
            values = values[:, :L, :, :]  # 截断数值序列
            keys = keys[:, :L, :, :]  # 截断键序列

        # 基于周期的依赖关系
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)  # 对查询进行快速傅里叶变换
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)  # 对键进行快速傅里叶变换
        res = q_fft * torch.conj(k_fft)  # 计算频域中的乘积
        corr = torch.fft.irfft(res, dim=-1)  # 进行逆傅里叶变换，得到相关性

        # 时间延迟聚合
        if self.training:  # 如果在训练阶段
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
            # 使用训练版聚合
        else:  # 否则
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
            # 使用推断版聚合

        if self.output_attention:  # 如果需要输出注意力权重
            return (V.contiguous(), corr.permute(0, 3, 1, 2))  # 返回聚合结果和相关性
        else:  # 否则
            return (V.contiguous(), None)  # 仅返回聚合结果，不返回相关性

    class AutoCorrelationLayer(nn.Module):  # 定义自相关层类，继承自 nn.Module
        def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
            # 初始化函数，设置类的参数和超参数
            super(AutoCorrelationLayer, self).__init__()  # 调用父类的初始化方法

            d_keys = d_keys or (d_model // n_heads)  # 如果未指定 d_keys，则使用 d_model / n_heads
            d_values = d_values or (d_model // n_heads)  # 如果未指定 d_values，则使用 d_model / n_heads

            self.inner_correlation = correlation  # 内部自相关机制
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # 查询投影层
            self.key_projection = nn.Linear(d_model, d_keys * n_heads)  # 键投影层
            self.value_projection = nn.Linear(d_model, d_values * n_heads)  # 数值投影层
            self.out_projection = nn.Linear(d_values * n_heads, d_model)  # 输出投影层
            self.n_heads = n_heads  # head 的数量

        def forward(self, queries, keys, values, attn_mask):  # 定义前向传播函数
            B, L, _ = queries.shape  # 获取查询的形状 (批量, 序列长度, 嵌入维度)
            _, S, _ = keys.shape  # 获取键的形状 (批量, 序列长度, 嵌入维度)
            H = self.n_heads  # 获取 head 的数量

            queries = self.query_projection(queries).view(B, L, H, -1)  # 对查询进行投影并调整维度
            keys = self.key_projection(keys).view(B, S, H, -1)  # 对键进行投影并调整维度
            values = self.value_projection(values).view(B, S, H, -1)  # 对数值进行投影并调整维度

            out, attn = self.inner_correlation(  # 调用内部自相关机制进行计算
                queries,
                keys,
                values,
                attn_mask
            )
            out = out.view(B, L, -1)  # 调整输出的维度

            return self.out_projection(out), attn  # 返回最终输出和注意力权重

