# 导入torch库，该库是用于深度学习的主要框架之一
import torch

# 定义TriangularCausalMask类，用于生成三角因果掩码
class TriangularCausalMask():
    # 类的初始化方法，初始化时需要指定批次大小、序列长度和设备（默认为CPU）
    def __init__(self, B, L, device="cpu"):
        """
        初始化TriangularCausalMask类。
        
        参数:
        - B: 批次大小
        - L: 序列长度
        - device: 设备类型（默认为"cpu"）
        """
        # 定义掩码的形状：[B, 1, L, L]
        mask_shape = [B, 1, L, L]
        
        # 在没有梯度跟踪的情况下，生成上三角掩码
        with torch.no_grad():
            # 使用torch.triu创建上三角矩阵，矩阵对角线以上的位置为True，其他位置为False
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    # 定义一个属性方法，返回生成的掩码
    @property
    def mask(self):
        """
        返回生成的三角因果掩码。
        
        :return: 生成的上三角掩码
        """
        return self._mask


# 定义ProbMask类，用于生成概率掩码
class ProbMask():
    # 类的初始化方法，初始化时需要指定批次大小、头的数量、序列长度、索引、评分矩阵和设备（默认为CPU）
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        初始化ProbMask类。
        
        参数:
        - B: 批次大小
        - H: 头的数量
        - L: 序列长度
        - index: 索引
        - scores: 评分矩阵
        - device: 设备类型（默认为"cpu"）
        """
        # 创建一个上三角矩阵掩码，形状为[L, scores的最后一维]，对角线以上为True
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        # 扩展掩码以匹配目标形状：[B, H, L, scores的最后一维]
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        
        # 根据批次大小和头数量来索引并生成最终的掩码矩阵
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        # 将掩码调整为与评分矩阵相同的形状
        self._mask = indicator.view(scores.shape).to(device)

    # 定义一个属性方法，返回生成的概率掩码
    @property
    def mask(self):
        """
        返回生成的概率掩码。
        
        :return: 生成的概率掩码矩阵
        """
        return self._mask
