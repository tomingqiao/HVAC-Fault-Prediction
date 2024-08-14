# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

# 导入PyTorch库，用于深度学习
import torch as t
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入NumPy库，用于科学计算
import numpy as np
# 导入pdb调试工具
import pdb

# 定义函数 divide_no_nan，该函数用于处理除法中可能出现的NaN和Inf值
def divide_no_nan(a, b):
    """
    进行 a/b 的除法操作，并将结果中的 NaN 或 Inf 替换为 0。

    参数:
    - a: 分子，可以是张量或数组
    - b: 分母，可以是张量或数组

    返回:
    - 返回除法结果，并将NaN和Inf替换为0的结果
    """
    # 执行除法运算
    result = a / b
    # 将结果中的 NaN 替换为 0
    result[result != result] = .0
    # 将结果中的 Inf 替换为 0
    result[result == np.inf] = .0
    # 返回处理后的结果
    return result

# 定义 mape_loss 类，用于计算MAPE损失（Mean Absolute Percentage Error）
class mape_loss(nn.Module):
    # 类的初始化方法，继承自nn.Module
    def __init__(self):
        super(mape_loss, self).__init__()  # 调用父类的初始化方法

    # 前向计算方法，用于计算损失值
    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        计算 MAPE 损失，定义参考链接: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        参数:
        - insample: 样本内数据，类型为 PyTorch 张量
        - freq: 频率参数，整数类型
        - forecast: 预测值，类型为 PyTorch 张量，形状为 (batch, time)
        - target: 目标值，类型为 PyTorch 张量，形状为 (batch, time)
        - mask: 掩码，类型为 PyTorch 张量，0/1 掩码，形状为 (batch, time)

        返回:
        - 损失值，类型为 PyTorch 浮点数
        """
        # 计算权重，mask与target的除法，并处理可能的NaN和Inf
        weights = divide_no_nan(mask, target)
        # 计算加权后的MAPE损失值
        return t.mean(t.abs((forecast - target) * weights))

# 定义 smape_loss 类，用于计算SMAPE损失（Symmetric Mean Absolute Percentage Error）
class smape_loss(nn.Module):
    # 类的初始化方法，继承自nn.Module
    def __init__(self):
        super(smape_loss, self).__init__()  # 调用父类的初始化方法

    # 前向计算方法，用于计算损失值
    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        计算 sMAPE 损失，定义参考链接: https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        参数:
        - insample: 样本内数据，类型为 PyTorch 张量
        - freq: 频率参数，整数类型
        - forecast: 预测值，类型为 PyTorch 张量，形状为 (batch, time)
        - target: 目标值，类型为 PyTorch 张量，形状为 (batch, time)
        - mask: 掩码，类型为 PyTorch 张量，0/1 掩码，形状为 (batch, time)

        返回:
        - 损失值，类型为 PyTorch 浮点数
        """
        # 计算SMAPE损失值，并处理可能的NaN和Inf
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)

# 定义 mase_loss 类，用于计算MASE损失（Mean Absolute Scaled Error）
class mase_loss(nn.Module):
    # 类的初始化方法，继承自nn.Module
    def __init__(self):
        super(mase_loss, self).__init__()  # 调用父类的初始化方法

    # 前向计算方法，用于计算损失值
    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        计算 MASE 损失，定义参考链接: "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        参数:
        - insample: 样本内数据，类型为 PyTorch 张量，形状为 (batch, time_i)
        - freq: 频率参数，整数类型
        - forecast: 预测值，类型为 PyTorch 张量，形状为 (batch, time_o)
        - target: 目标值，类型为 PyTorch 张量，形状为 (batch, time_o)
        - mask: 掩码，类型为 PyTorch 张量，0/1 掩码，形状为 (batch, time_o)

        返回:
        - 损失值，类型为 PyTorch 浮点数
        """
        # 计算样本内的误差平均值
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        # 计算掩码与误差平均值的除法结果，并处理可能的NaN和Inf
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        # 计算并返回MASE损失值
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

