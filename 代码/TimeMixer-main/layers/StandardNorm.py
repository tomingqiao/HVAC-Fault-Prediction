import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

# 定义 Normalize 类，继承自 nn.Module
class Normalize(nn.Module):
    # 初始化函数，定义类的基本参数
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        初始化 Normalize 类
        :param num_features: 特征或通道的数量
        :param eps: 一个用于数值稳定性的值
        :param affine: 如果为 True，RevIN 将具有可学习的仿射参数
        :param subtract_last: 如果为 True，将从最后一个元素减去
        :param non_norm: 如果为 True，不执行归一化操作
        """
        super(Normalize, self).__init__()  # 调用父类的初始化方法
        self.num_features = num_features  # 设置特征或通道的数量
        self.eps = eps  # 设置用于数值稳定性的值
        self.affine = affine  # 是否启用仿射变换
        self.subtract_last = subtract_last  # 是否从最后一个元素减去
        self.non_norm = non_norm  # 是否禁用归一化操作
        if self.affine:
            self._init_params()  # 如果启用了仿射变换，则初始化仿射参数

    # 前向传播函数，根据模式选择归一化或反归一化
    def forward(self, x, mode: str):
        """
        前向传播函数
        :param x: 输入的张量
        :param mode: 选择 'norm' 进行归一化，或 'denorm' 进行反归一化
        :return: 归一化或反归一化后的张量
        """
        if mode == 'norm':
            self._get_statistics(x)  # 获取输入张量的统计信息（均值和标准差）
            x = self._normalize(x)  # 执行归一化操作
        elif mode == 'denorm':
            x = self._denormalize(x)  # 执行反归一化操作
        else:
            raise NotImplementedError  # 如果模式不支持，则引发未实现错误
        return x  # 返回处理后的张量

    # 初始化仿射变换的参数
    def _init_params(self):
        # 初始化 RevIN 参数，形状为 (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))  # 初始化仿射权重为 1
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))  # 初始化仿射偏差为 0

    # 获取输入张量的统计信息（均值和标准差）
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))  # 设置要减少的维度，通常是所有除了批次和特征维度的维度
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)  # 如果减去最后一个元素，则获取最后一元素
        else:
            # 否则计算张量在指定维度上的均值，并使用 detach() 函数使其不参与梯度计算
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # 计算方差的平方根（标准差），并添加 eps 以避免数值不稳定
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    # 执行归一化操作
    def _normalize(self, x):
        if self.non_norm:
            return x  # 如果禁用了归一化，直接返回原输入张量
        if self.subtract_last:
            x = x - self.last  # 如果减去最后一个元素，则从每个元素中减去最后一元素
        else:
            x = x - self.mean  # 否则从每个元素中减去均值
        x = x / self.stdev  # 将每个元素除以标准差，完成归一化
        if self.affine:
            x = x * self.affine_weight  # 如果启用了仿射变换，乘以仿射权重
            x = x + self.affine_bias  # 然后加上仿射偏差
        return x  # 返回归一化后的张量

    # 执行反归一化操作
    def _denormalize(self, x):
        if self.non_norm:
            return x  # 如果禁用了归一化，直接返回原输入张量
        if self.affine:
            x = x - self.affine_bias  # 如果启用了仿射变换，首先减去仿射偏差
            x = x / (self.affine_weight + self.eps * self.eps)  # 然后除以仿射权重，并考虑数值稳定性
        x = x * self.stdev  # 将每个元素乘以标准差，完成反归一化
        if self.subtract_last:
            x = x + self.last  # 如果减去最后一个元素，则加回最后一元素
        else:
            x = x + self.mean  # 否则加回均值
        return x  # 返回反归一化后的张量
