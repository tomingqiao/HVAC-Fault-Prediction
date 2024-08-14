import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式接口
from torch.nn.utils import weight_norm  # 导入PyTorch的权重归一化工具
import math  # 导入数学库

# 位置编码类，用于为输入序列添加位置信息
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()  # 调用父类的初始化方法
        # 计算位置编码，一次性在log空间中完成
        pe = torch.zeros(max_len, d_model).float()  # 初始化位置编码张量
        pe.require_grad = False  # 设置不需要梯度计算

        position = torch.arange(0, max_len).float().unsqueeze(1)  # 生成位置索引
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()  # 计算位置编码的除数

        pe[:, 0::2] = torch.sin(position * div_term)  # 在偶数维度上应用sin函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 在奇数维度上应用cos函数

        pe = pe.unsqueeze(0)  # 扩展位置编码张量以匹配批次维度
        self.register_buffer('pe', pe)  # 注册位置编码为模型的持久性缓存

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # 返回与输入序列长度匹配的部分位置编码


# Token嵌入类，用于将输入的token进行嵌入
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()  # 调用父类的初始化方法
        padding = 1 if torch.__version__ >= '1.5.0' else 2  # 根据PyTorch版本选择填充大小
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)  # 定义1D卷积层

        # 初始化卷积层的权重
        for m in self.modules():  # 遍历所有模块
            if isinstance(m, nn.Conv1d):  # 如果是卷积层
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')  # 使用Kaiming正态分布初始化权重

    def forward(self, x):
        # 调整输入张量的维度，使其适应1D卷积层
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x  # 返回嵌入后的张量


# 固定嵌入类，用于将输入固定为一个预定义的嵌入
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()  # 调用父类的初始化方法

        w = torch.zeros(c_in, d_model).float()  # 初始化嵌入矩阵
        w.require_grad = False  # 设置不需要梯度计算

        position = torch.arange(0, c_in).float().unsqueeze(1)  # 生成位置索引
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()  # 计算位置编码的除数

        w[:, 0::2] = torch.sin(position * div_term)  # 在偶数维度上应用sin函数
        w[:, 1::2] = torch.cos(position * div_term)  # 在奇数维度上应用cos函数

        self.emb = nn.Embedding(c_in, d_model)  # 定义嵌入层
        self.emb.weight = nn.Parameter(w, requires_grad=False)  # 将预定义的嵌入矩阵作为参数

    def forward(self, x):
        return self.emb(x).detach()  # 返回固定嵌入并从计算图中分离


# 时间嵌入类，用于将时间信息嵌入到输入序列中
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()  # 调用父类的初始化方法

        # 定义不同时间维度的大小
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # 根据嵌入类型选择嵌入层
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)  # 定义分钟嵌入层
        self.hour_embed = Embed(hour_size, d_model)  # 定义小时嵌入层
        self.weekday_embed = Embed(weekday_size, d_model)  # 定义星期几嵌入层
        self.day_embed = Embed(day_size, d_model)  # 定义天嵌入层
        self.month_embed = Embed(month_size, d_model)  # 定义月份嵌入层

    def forward(self, x):
        x = x.long()  # 将输入转换为long类型
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.  # 获取分钟嵌入，如果有定义的话
        hour_x = self.hour_embed(x[:, :, 3])  # 获取小时嵌入
        weekday_x = self.weekday_embed(x[:, :, 2])  # 获取星期几嵌入
        day_x = self.day_embed(x[:, :, 1])  # 获取天嵌入
        month_x = self.month_embed(x[:, :, 0])  # 获取月份嵌入

        return hour_x + weekday_x + day_x + month_x + minute_x  # 返回所有时间嵌入的和


# 时间特征嵌入类，用于将时间特征嵌入到输入序列中
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()  # 调用父类的初始化方法

        # 定义不同频率对应的输入维度
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]  # 根据频率选择输入维度
        self.embed = nn.Linear(d_inp, d_model, bias=False)  # 定义线性层作为嵌入层

    def forward(self, x):
        return self.embed(x)  # 返回时间特征嵌入后的张量


# 数据嵌入类，用于将多种嵌入方式组合在一起
class DataEmbedding(nn.Module):
    # 初始化函数
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()  # 调用父类的初始化方法
        
        # 保存输入通道数和模型维度
        self.c_in = c_in  
        self.d_model = d_model  
        
        # 初始化值嵌入层
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # 初始化位置嵌入层
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        # 根据嵌入类型选择时间嵌入层
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        
        # 初始化Dropout层
        self.dropout = nn.Dropout(p=dropout)

    # 前向传播函数
    def forward(self, x, x_mark):
        _, _, N = x.size()  # 获取输入张量的维度信息
        
        # 如果输入的第三个维度等于输入通道数
        if N == self.c_in:
            # 如果没有时间标记
            if x_mark is None:
                x = self.value_embedding(x) + self.position_embedding(x)  # 仅进行值嵌入和位置嵌入
            else:
                x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        
        # 如果输入的第三个维度等于模型维度
        elif N == self.d_model:
            if x_mark is None:
                x = x + self.position_embedding(x)  # 仅进行位置嵌入
            else:
                x = x + self.temporal_embedding(x_mark) + self.position_embedding(x)  # 进行时间嵌入和位置嵌入

        return self.dropout(x)  # 返回经过Dropout处理后的张量


# 多尺度数据嵌入类
class DataEmbedding_ms(nn.Module):
    # 初始化函数
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_ms, self).__init__()  # 调用父类的初始化方法
        
        # 初始化值嵌入层，注意此处输入通道数固定为1
        self.value_embedding = TokenEmbedding(c_in=1, d_model=d_model)
        
        # 初始化位置嵌入层
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        # 根据嵌入类型选择时间嵌入层
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        
        # 初始化Dropout层
        self.dropout = nn.Dropout(p=dropout)

    # 前向传播函数
    def forward(self, x, x_mark):
        B, T, N = x.shape  # 获取输入张量的维度信息
        
        # 对输入张量进行变形和嵌入
        x1 = self.value_embedding(x.reshape(0, 2, 1).reshape(B * N, T).unsqueeze(-1)).reshape(B, N, T, -1).permute(0, 2, 1, 3)
        
        # 如果没有时间标记
        if x_mark is None:
            x = x1  # 仅使用值嵌入
        else:
            x = x1 + self.temporal_embedding(x_mark)  # 使用值嵌入和时间嵌入
        
        return self.dropout(x)  # 返回经过Dropout处理后的张量


# 不带位置嵌入的数据嵌入类
class DataEmbedding_wo_pos(nn.Module):
    # 初始化函数
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()  # 调用父类的初始化方法
        
        # 初始化值嵌入层
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # 初始化位置嵌入层
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        # 根据嵌入类型选择时间嵌入层
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        
        # 初始化Dropout层
        self.dropout = nn.Dropout(p=dropout)

    # 前向传播函数
    def forward(self, x, x_mark):
        # 如果x为空且x_mark不为空
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)  # 返回时间嵌入
        
        # 如果x_mark为空
        if x_mark is None:
            x = self.value_embedding(x)  # 仅进行值嵌入
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)  # 进行值嵌入和时间嵌入
        
        return self.dropout(x)  # 返回经过Dropout处理后的张量


# Crossformer模型中的补丁嵌入类
class PatchEmbedding_crossformer(nn.Module):
    # 初始化函数
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding_crossformer, self).__init__()  # 调用父类的初始化方法
        
        # 初始化补丁长度、步幅和填充层
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        
        # 初始化值嵌入层，将特征向量投影到d维向量空间
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        # 初始化位置嵌入层
        self.position_embedding = PositionalEmbedding(d_model)
        
        # 初始化Dropout层
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数
    def forward(self, x):
        n_vars = x.shape[1]  # 获取输入张量的第二维度（特征数）
        
        x = self.padding_patch_layer(x)  # 对输入张量进行填充
        
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 对输入张量进行展开
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # 对展开后的张量进行变形
        
        # 进行值嵌入和位置嵌入
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x), n_vars  # 返回经过Dropout处理后的张量和特征数


# 普通补丁嵌入类
class PatchEmbedding(nn.Module):
    # 初始化函数
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()  # 调用父类的初始化方法
        
        # 初始化补丁长度、步幅和填充层
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        
        # 初始化值嵌入层，将特征向量投影到d维向量空间
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        
        # 初始化位置嵌入层
        self.position_embedding = PositionalEmbedding(d_model)
        
        # 初始化Dropout层
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数
    def forward(self, x):
        n_vars = x.shape[1]  # 获取输入张量的第二维度（特征数）
        
        x = self.padding_patch_layer(x)  # 对输入张量进行填充
        
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 对输入张量进行展开
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # 对展开后的张量进行变形
        
        # 进行值嵌入和位置嵌入
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x), n_vars  # 返回经过Dropout处理后的张量和特征数
