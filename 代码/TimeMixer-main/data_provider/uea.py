import os
import numpy as np
import pandas as pd
import torch


def collate_fn(data, max_len=None):
    """
    从 (X, mask) 元组列表中构建小批量张量。掩码输入。
    
    Args:
        data: 长度为 batch_size 的元组列表 (X, y)。
            - X: 形状为 (seq_length, feat_dim) 的 torch 张量；序列长度可变。
            - y: 形状为 (num_labels,) 的 torch 张量：类别索引或数值目标
                （分别用于分类或回归）。如果是多任务模型，num_labels > 1
        max_len: 全局固定序列长度。用于需要固定长度输入的架构，
            其中批次长度不能动态变化。较长的序列会被截断，较短的会用0填充。

    Returns:
        X: 形状为 (batch_size, padded_length, feat_dim) 的 torch 张量，表示掩码后的特征（输入）
        targets: 形状为 (batch_size, num_labels) 的 torch 张量，表示未掩码的特征（输出）
        padding_masks: 形状为 (batch_size, padded_length) 的布尔 torch 张量
            0 表示需要预测的掩码值，1 表示未受影响的“活动”特征值
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # 堆叠和填充特征和掩码（将2D张量转换为3D张量，即添加批次维度）
    lengths = [X.shape[0] for X in features]  # 每个时间序列的原始序列长度
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) 布尔张量，“1”表示保留

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    用于掩码填充位置：从序列长度张量创建一个 (batch_size, max_len) 的布尔掩码，
    其中 1 表示保留当前位置（时间步）的元素
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # 这个技巧使用了“或”运算符的非布尔类型重载
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))  # 创建掩码，1 表示该位置有值，0 表示填充


class Normalizer(object):
    """
    对整个 DataFrame 进行标准化（所有时间步）。与逐样本标准化不同。
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: 选择以下类型：
                "standardization", "minmax": 对整个 DataFrame 进行标准化（所有时间步）
                "per_sample_std", "per_sample_minmax": 单独对每个样本进行标准化（即仅对其自身的行）
            mean, std, min_val, max_val: 可选的（num_feat,）预计算值系列
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: 输入的 DataFrame
        Returns:
            df: 标准化后的 DataFrame
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()  # 计算均值
                self.std = df.std()  # 计算标准差
            return (df - self.mean) / (self.std + np.finfo(float).eps)  # 标准化，防止除以零

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()  # 计算最大值
                self.min_val = df.min()  # 计算最小值
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)  # Min-Max 标准化

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')  # 逐样本标准化（标准差）

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)  # 逐样本标准化（Min-Max）

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))  # 如果方法未实现，抛出异常


def interpolate_missing(y):
    """
    使用线性插值替换 pd.Series `y` 中的 NaN 值
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')  # 使用线性插值，向两个方向插值
    return y


def subsample(y, limit=256, factor=2):
    """
    如果给定的序列长度超过 `limit`，则通过指定的整数因子返回下采样后的序列
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)  # 每隔 factor 采样一次，并重置索引
    return y
