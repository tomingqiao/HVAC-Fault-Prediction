import torch
import torch.nn as nn
import torch.nn.functional as F

class my_Layernorm(nn.Module):  # 自定义层归一化类
    """
    针对季节性部分特殊设计的层归一化
    """

    def __init__(self, channels):  # 初始化函数，传入通道数
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)  # 初始化标准的LayerNorm

    def forward(self, x):  # 前向传播函数
        x_hat = self.layernorm(x)  # 对输入进行层归一化
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)  # 计算沿通道维度的均值并重复扩展以匹配输入维度
        return x_hat - bias  # 返回去除偏差后的结果

class moving_avg(nn.Module):  # 滑动平均类
    """
    滑动平均模块，用于突出时间序列的趋势部分
    """

    def __init__(self, kernel_size, stride):  # 初始化函数，传入核大小和步幅
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size  # 保存核大小
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)  # 使用1D平均池化实现滑动平均

    def forward(self, x):  # 前向传播函数
        # 对时间序列的两端进行填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # 在前端进行填充
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # 在后端进行填充
        x = torch.cat([front, x, end], dim=1)  # 拼接填充后的序列
        x = self.avg(x.permute(0, 2, 1))  # 调整维度顺序后进行平均池化
        x = x.permute(0, 2, 1)  # 再次调整维度顺序以匹配输入
        return x  # 返回滑动平均后的序列

class series_decomp(nn.Module):  # 序列分解类
    """
    序列分解模块
    """

    def __init__(self, kernel_size):  # 初始化函数，传入核大小
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)  # 初始化滑动平均模块，步幅为1

    def forward(self, x):  # 前向传播函数
        moving_mean = self.moving_avg(x)  # 计算滑动平均，即时间序列的趋势部分
        res = x - moving_mean  # 计算残差，即时间序列的季节性部分
        return res, moving_mean  # 返回季节性部分和趋势部分

class series_decomp_multi(nn.Module):  # 多序列分解类
    """
    FEDformer中的多序列分解模块
    """

    def __init__(self, kernel_size):  # 初始化函数，传入多种核大小的列表
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size  # 保存核大小列表
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]  # 为每个核大小初始化一个序列分解模块

    def forward(self, x):  # 前向传播函数
        moving_mean = []  # 用于存储各个核大小对应的趋势部分
        res = []  # 用于存储各个核大小对应的季节性部分
        for func in self.series_decomp:  # 对每个分解模块进行操作
            sea, moving_avg = func(x)  # 计算季节性部分和趋势部分
            moving_mean.append(moving_avg)  # 保存趋势部分
            res.append(sea)  # 保存季节性部分

        sea = sum(res) / len(res)  # 对所有季节性部分取平均，得到最终的季节性部分
        moving_mean = sum(moving_mean) / len(moving_mean)  # 对所有趋势部分取平均，得到最终的趋势部分
        return sea, moving_mean  # 返回季节性部分和趋势部分

class EncoderLayer(nn.Module):  # 编码器层类
    """
    Autoformer编码器层，具有渐进分解架构
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):  # 初始化函数
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # 如果未指定d_ff，则设置为4倍的d_model
        self.attention = attention  # 注意力机制模块
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)  # 1D卷积，用于增加维度
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)  # 1D卷积，用于恢复维度
        self.decomp1 = series_decomp(moving_avg)  # 初始化第一个序列分解模块
        self.decomp2 = series_decomp(moving_avg)  # 初始化第二个序列分解模块
        self.dropout = nn.Dropout(dropout)  # Dropout层，用于防止过拟合
        self.activation = F.relu if activation == "relu" else F.gelu  # 根据激活函数名称选择激活函数

    def forward(self, x, attn_mask=None):  # 前向传播函数
        new_x, attn = self.attention(  # 通过注意力机制模块
            x, x, x,  # 查询、键和值都为输入x
            attn_mask=attn_mask  # 可选的注意力掩码
        )
        x = x + self.dropout(new_x)  # 残差连接并经过Dropout
        x, _ = self.decomp1(x)  # 通过第一个序列分解模块，仅保留残差部分（去掉趋势）
        y = x  # 复制x以进行卷积操作
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 维度转换，激活并经过第一个卷积层
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # 维度恢复并经过第二个卷积层
        res, _ = self.decomp2(x + y)  # 第二次序列分解，得到最终的残差部分
        return res, attn  # 返回残差部分和注意力权重



class Encoder(nn.Module):
    """
    Autoformer编码器类
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        初始化函数
        
        参数：
        - attn_layers: 注意力层列表
        - conv_layers: 卷积层列表，可选
        - norm_layer: 归一化层，可选
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # 将注意力层包装为ModuleList以便于管理
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None  # 如果卷积层不为空，将其包装为ModuleList
        self.norm = norm_layer  # 保存归一化层

    def forward(self, x, attn_mask=None):
        """
        前向传播函数
        
        参数：
        - x: 输入数据张量
        - attn_mask: 注意力掩码，可选
        
        返回：
        - x: 编码后的输出
        - attns: 注意力权重列表
        """
        attns = []  # 初始化注意力权重列表
        if self.conv_layers is not None:  # 如果存在卷积层
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):  # 遍历注意力层和卷积层
                x, attn = attn_layer(x, attn_mask=attn_mask)  # 通过注意力层计算
                x = conv_layer(x)  # 通过卷积层计算
                attns.append(attn)  # 保存注意力权重
            x, attn = self.attn_layers[-1](x)  # 处理最后一个注意力层
            attns.append(attn)  # 保存最后一个注意力权重
        else:  # 如果没有卷积层
            for attn_layer in self.attn_layers:  # 仅遍历注意力层
                x, attn = attn_layer(x, attn_mask=attn_mask)  # 通过注意力层计算
                attns.append(attn)  # 保存注意力权重

        if self.norm is not None:  # 如果存在归一化层
            x = self.norm(x)  # 对输出进行归一化

        return x, attns  # 返回编码后的输出和注意力权重


class DecoderLayer(nn.Module):
    """
    Autoformer解码器层类，具有渐进分解架构
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        """
        初始化函数
        
        参数：
        - self_attention: 自注意力层
        - cross_attention: 交叉注意力层
        - d_model: 模型维度
        - c_out: 输出通道数
        - d_ff: 前向传播的隐藏层维度，可选
        - moving_avg: 滑动平均窗口大小
        - dropout: Dropout比例
        - activation: 激活函数类型
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # 如果d_ff未指定，默认设为4倍的d_model
        self.self_attention = self_attention  # 自注意力层
        self.cross_attention = cross_attention  # 交叉注意力层
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)  # 卷积层1，用于升维
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)  # 卷积层2，用于降维
        self.decomp1 = series_decomp(moving_avg)  # 初始化第一个序列分解模块
        self.decomp2 = series_decomp(moving_avg)  # 初始化第二个序列分解模块
        self.decomp3 = series_decomp(moving_avg)  # 初始化第三个序列分解模块
        self.dropout = nn.Dropout(dropout)  # Dropout层，用于防止过拟合
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)  # 投影层，用于调整维度
        self.activation = F.relu if activation == "relu" else F.gelu  # 选择激活函数

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        前向传播函数
        
        参数：
        - x: 输入数据张量
        - cross: 交叉注意力的输入张量
        - x_mask: 自注意力掩码，可选
        - cross_mask: 交叉注意力掩码，可选
        
        返回：
        - x: 解码后的输出
        - residual_trend: 残余趋势
        """
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask  # 自注意力计算
        )[0])
        x, trend1 = self.decomp1(x)  # 第一次序列分解
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask  # 交叉注意力计算
        )[0])
        x, trend2 = self.decomp2(x)  # 第二次序列分解
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 维度变换，激活并通过第一个卷积层
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # 维度恢复并通过第二个卷积层
        x, trend3 = self.decomp3(x + y)  # 第三次序列分解

        residual_trend = trend1 + trend2 + trend3  # 计算残余趋势
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)  # 通过投影层进行维度调整
        return x, residual_trend  # 返回解码后的输出和残余趋势


class Decoder(nn.Module):
    """
    Autoformer解码器类
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        """
        初始化函数
        
        参数：
        - layers: 解码层列表
        - norm_layer: 归一化层，可选
        - projection: 投影层，可选
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)  # 将解码层包装为ModuleList以便于管理
        self.norm = norm_layer  # 保存归一化层
        self.projection = projection  # 保存投影层

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        """
        前向传播函数
        
        参数：
        - x: 输入数据张量
        - cross: 交叉注意力的输入张量
        - x_mask: 自注意力掩码，可选
        - cross_mask: 交叉注意力掩码，可选
        - trend: 输入的趋势张量
        
        返回：
        - x: 解码后的输出
        - trend: 更新后的趋势张量
        """
        for layer in self.layers:  # 遍历每个解码层
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)  # 通过解码层计算
            trend = trend + residual_trend  # 累加残余趋势

        if self.norm is not None:  # 如果存在归一化层
            x = self.norm(x)  # 对输出进行归一化

        if self.projection is not None:  # 如果存在投影层
            x = self.projection(x)  # 对输出进行投影
        return x, trend  # 返回解码后的输出和趋势张量