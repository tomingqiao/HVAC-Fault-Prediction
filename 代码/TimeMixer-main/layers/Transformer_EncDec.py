import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性模块，包含激活函数等

# 定义卷积层类 ConvLayer，继承自 nn.Module
class ConvLayer(nn.Module):
    # 初始化函数，定义类的基本参数和层
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()  # 调用父类的初始化方法
        # 定义一维卷积层
        self.downConv = nn.Conv1d(in_channels=c_in,  # 输入通道数
                                  out_channels=c_in,  # 输出通道数，保持不变
                                  kernel_size=3,  # 卷积核大小为3
                                  padding=2,  # 填充大小为2
                                  padding_mode='circular')  # 使用循环填充模式
        self.norm = nn.BatchNorm1d(c_in)  # 定义一维批量归一化层
        self.activation = nn.ELU()  # 使用 ELU 作为激活函数
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 定义最大池化层

    # 前向传播函数，定义前向计算过程
    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))  # 先对输入张量进行维度转换，然后进行卷积运算
        x = self.norm(x)  # 进行批量归一化
        x = self.activation(x)  # 通过激活函数
        x = self.maxPool(x)  # 进行最大池化
        x = x.transpose(1, 2)  # 将张量的维度转换回原始形状
        return x  # 返回处理后的张量


# 定义编码器层类 EncoderLayer，继承自 nn.Module
class EncoderLayer(nn.Module):
    # 初始化函数，定义编码器层的基本参数和层
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self.__init__())  # 调用父类的初始化方法
        d_ff = d_ff or 4 * d_model  # 如果未定义 d_ff，默认使用 4 倍 d_model
        self.attention = attention  # 设置自注意力机制
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)  # 定义第一个一维卷积层
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)  # 定义第二个一维卷积层
        self.norm1 = nn.LayerNorm(d_model)  # 定义第一层归一化层
        self.norm2 = nn.LayerNorm(d_model)  # 定义第二层归一化层
        self.dropout = nn.Dropout(dropout)  # 定义 Dropout 层
        self.activation = F.relu if activation == "relu" else F.gelu  # 选择激活函数

    # 前向传播函数，定义前向计算过程
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,  # 自注意力机制的输入，使用相同的输入作为查询、键和值
            attn_mask=attn_mask,  # 可选的注意力掩码
            tau=tau, delta=delta  # 可选的超参数，用于调整注意力机制
        )
        x = x + self.dropout(new_x)  # 添加残差连接，并应用 Dropout

        y = x = self.norm1(x)  # 进行层归一化
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 卷积、激活和 Dropout 操作
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # 反转维度，卷积和 Dropout 操作

        return self.norm2(x + y), attn  # 返回归一化后的输出和注意力


# 定义编码器类 Encoder，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化函数，定义编码器的基本参数和层
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()  # 调用父类的初始化方法
        self.attn_layers = nn.ModuleList(attn_layers)  # 将注意力层列表转换为模块列表
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None  # 如果存在卷积层，则转换为模块列表
        self.norm = norm_layer  # 定义归一化层

    # 前向传播函数，定义前向计算过程
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []  # 用于存储注意力的列表
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None  # 在第一个层使用 delta，其它层则不使用
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)  # 计算注意力和卷积
                x = conv_layer(x)  # 进行卷积操作
                attns.append(attn)  # 将注意力添加到列表中
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)  # 处理最后一层
            attns.append(attn)  # 添加注意力到列表
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)  # 处理每一层
                attns.append(attn)  # 添加注意力到列表

        if self.norm is not None:
            x = self.norm(x)  # 进行归一化

        return x, attns  # 返回编码后的输出和所有注意力


# 定义解码器层类 DecoderLayer，继承自 nn.Module
class DecoderLayer(nn.Module):
    # 初始化函数，定义解码器层的基本参数和层
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()  # 调用父类的初始化方法
        d_ff = d_ff or 4 * d_model  # 如果未定义 d_ff，默认使用 4 倍 d_model
        self.self_attention = self_attention  # 设置自注意力机制
        self.cross_attention = cross_attention  # 设置交叉注意力机制
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)  # 定义第一个一维卷积层
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)  # 定义第二个一维卷积层
        self.norm1 = nn.LayerNorm(d_model)  # 定义第一层归一化层
        self.norm2 = nn.LayerNorm(d_model)  # 定义第二层归一化层
        self.norm3 = nn.LayerNorm(d_model)  # 定义第三层归一化层
        self.dropout = nn.Dropout(dropout)  # 定义 Dropout 层
        self.activation = F.relu if activation == "relu" else F.gelu  # 选择激活函数

    # 前向传播函数，定义前向计算过程
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,  # 自注意力机制的输入，使用相同的输入作为查询、键和值
            attn_mask=x_mask,  # 可选的注意力掩码
            tau=tau, delta=None  # 可选的超参数，用于调整注意力机制
        )[0])
        x = self.norm1(x)  # 第一层归一化

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,  # 交叉注意力机制，使用交叉的输入作为查询、键和值
            attn_mask=cross_mask,  # 可选的交叉注意力掩码
            tau=tau, delta=delta  # 可选的超参数，用于调整注意力机制
        )[0])

        y = x = self.norm2(x)  # 第二层归一化
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 卷积、激活和 Dropout 操作
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # 反转维度，卷积和 Dropout 操作

        return self.norm3(x + y)  # 返回归一化后的输出


# 定义解码器类 Decoder，继承自 nn.Module
class Decoder(nn.Module):
    # 初始化函数，定义解码器的基本参数和层
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()  # 调用父类的初始化方法
        self.layers = nn.ModuleList(layers)  # 将解码层列表转换为模块列表
        self.norm = norm_layer  # 定义归一化层
        self.projection = projection  # 定义投影层

    # 前向传播函数，定义前向计算过程
    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:  # 遍历所有解码层
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)  # 进行解码操作

        if self.norm is not None:
            x = self.norm(x)  # 进行归一化

        if self.projection is not None:
            x = self.projection(x)  # 进行投影操作
        return x  # 返回解码后的输出
