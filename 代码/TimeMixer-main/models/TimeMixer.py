import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性模块，包含激活函数等
from layers.Autoformer_EncDec import series_decomp  # 从自定义模块中导入时间序列分解函数
from layers.Embed import DataEmbedding_wo_pos  # 从自定义模块中导入数据嵌入类（无位置编码）
from layers.StandardNorm import Normalize  # 从自定义模块中导入标准化类

# 定义时间序列分解的类 DFT_series_decomp，继承自 nn.Module
class DFT_series_decomp(nn.Module):
    """
    时间序列分解块
    """

    # 初始化函数，定义类的基本参数
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()  # 调用父类的初始化方法
        self.top_k = top_k  # 设置选取频率的最高 k 值

    # 前向传播函数，定义前向计算过程
    def forward(self, x):
        xf = torch.fft.rfft(x)  # 对输入 x 进行快速傅里叶变换，得到频域表示
        freq = abs(xf)  # 计算频域表示的幅值
        freq[0] = 0  # 将直流分量置零，以去除趋势
        top_k_freq, top_list = torch.topk(freq, 5)  # 选取频率最高的 k 个分量
        xf[freq <= top_k_freq.min()] = 0  # 过滤掉不在前 k 大频率分量中的其他分量
        x_season = torch.fft.irfft(xf)  # 对保留的频率分量进行逆傅里叶变换，得到季节性分量
        x_trend = x - x_season  # 通过减去季节性分量，得到趋势分量
        return x_season, x_trend  # 返回季节性分量和趋势分量

# 定义多尺度季节性混合类 MultiScaleSeasonMixing，继承自 nn.Module
class MultiScaleSeasonMixing(nn.Module):
    """
    自底向上的季节性模式混合
    """

    # 初始化函数，定义类的基本参数和层
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()  # 调用父类的初始化方法

        # 定义降采样层的列表
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(  # 定义一个顺序容器，包含线性层和激活函数
                    torch.nn.Linear(  # 线性层
                        configs.seq_len // (configs.down_sampling_window ** i),  # 输入的维度，随着降采样窗口的增大而减小
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),  # 输出的维度，继续减小
                    ),
                    nn.GELU(),  # GELU 激活函数
                    torch.nn.Linear(  # 线性层
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.down_sampling_layers)  # 对于每个降采样层进行迭代
            ]
        )

    # 前向传播函数，定义前向计算过程
    def forward(self, season_list):

        # 高到低的混合过程
        out_high = season_list[0]  # 初始的高频季节性模式
        out_low = season_list[1]  # 初始的低频季节性模式
        out_season_list = [out_high.permute(0, 2, 1)]  # 将高频模式转置并加入输出列表

        # 对季节性模式列表进行逐层处理
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)  # 对高频模式进行降采样
            out_low = out_low + out_low_res  # 将降采样后的高频模式加到低频模式中
            out_high = out_low  # 更新高频模式为当前的低频模式
            if i + 2 <= len(season_list) - 1:  # 判断是否存在下一个低频季节性模式
                out_low = season_list[i + 2]  # 更新低频模式为下一层的低频模式
            out_season_list.append(out_high.permute(0, 2, 1))  # 将处理后的高频模式加入输出列表

        return out_season_list  # 返回混合后的季节性模式列表

# 定义多尺度趋势混合类 MultiScaleTrendMixing，继承自 nn.Module
class MultiScaleTrendMixing(nn.Module):
    """
    自顶向下的趋势模式混合
    """

    # 初始化函数，定义类的基本参数和层
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()  # 调用父类的初始化方法

        # 定义上采样层的列表
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(  # 定义一个顺序容器，包含线性层和激活函数
                    torch.nn.Linear(  # 线性层
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),  # 输入的维度，随着上采样窗口的减小而增大
                        configs.seq_len // (configs.down_sampling_window ** i),  # 输出的维度，继续增大
                    ),
                    nn.GELU(),  # GELU 激活函数
                    torch.nn.Linear(  # 线性层
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))  # 逆序迭代降采样层
            ])

    # 前向传播函数，定义前向计算过程
    def forward(self, trend_list):

        # 低到高的混合过程
        trend_list_reverse = trend_list.copy()  # 复制趋势列表
        trend_list_reverse.reverse()  # 逆序趋势列表
        out_low = trend_list_reverse[0]  # 初始的低频趋势模式
        out_high = trend_list_reverse[1]  # 初始的高频趋势模式
        out_trend_list = [out_low.permute(0, 2, 1)]  # 将低频模式转置并加入输出列表

        # 对趋势模式列表进行逐层处理
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)  # 对低频模式进行上采样
            out_high = out_high + out_high_res  # 将上采样后的低频模式加到高频模式中
            out_low = out_high  # 更新低频模式为当前的高频模式
            if i + 2 <= len(trend_list_reverse) - 1:  # 判断是否存在下一个高频趋势模式
                out_high = trend_list_reverse[i + 2]  # 更新高频模式为下一层的高频模式
            out_trend_list.append(out_low.permute(0, 2, 1))  # 将处理后的低频模式加入输出列表

        out_trend_list.reverse()  # 逆序输出趋势列表，使其按高到低的顺序排列
        return out_trend_list  # 返回混合后的趋势模式列表

# 定义可分解的历史混合类 PastDecomposableMixing，继承自 nn.Module
class PastDecomposableMixing(nn.Module):
    # 初始化函数，定义类的基本参数和层
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()  # 调用父类的初始化方法
        self.seq_len = configs.seq_len  # 序列长度
        self.pred_len = configs.pred_len  # 预测长度
        self.down_sampling_window = configs.down_sampling_window  # 降采样窗口大小

        self.layer_norm = nn.LayerNorm(configs.d_model)  # 定义层归一化
        self.dropout = nn.Dropout(configs.dropout)  # 定义 Dropout 层
        self.channel_independence = configs.channel_independence  # 是否进行通道独立性处理

        # 根据分解方法选择相应的分解方式
        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)  # 使用移动平均进行分解
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)  # 使用傅里叶变换进行分解
        else:
            raise ValueError('decompsition is error')  # 如果分解方法不支持，抛出异常

        # 如果不进行通道独立性处理，定义交叉层
        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),  # 线性层
                nn.GELU(),  # GELU 激活函数
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),  # 线性层
            )

        # 定义季节性模式的多尺度混合层
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # 定义趋势模式的多尺度混合层
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        # 定义输出的交叉层
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),  # 线性层
            nn.GELU(),  # GELU 激活函数
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),  # 线性层
        )

    # 前向传播函数，定义前向计算过程
    def forward(self, x_list):
        length_list = []  # 用于存储每个输入序列的长度
        for x in x_list:
            _, T, _ = x.size()  # 获取序列长度 T
            length_list.append(T)  # 将序列长度添加到列表中

        # 对每个输入序列进行分解，得到季节性和趋势部分
        season_list = []  # 季节性列表
        trend_list = []  # 趋势列表
        for x in x_list:
            season, trend = self.decompsition(x)  # 调用分解函数，分解为季节性和趋势
            if self.channel_independence == 0:  # 如果不进行通道独立性处理，应用交叉层
                season = self.cross_layer(season)  # 处理季节性分量
                trend = self.cross_layer(trend)  # 处理趋势分量
            season_list.append(season.permute(0, 2, 1))  # 将季节性分量转置并添加到列表中
            trend_list.append(trend.permute(0, 2, 1))  # 将趋势分量转置并添加到列表中

        # 自底向上的季节性混合
        out_season_list = self.mixing_multi_scale_season(season_list)
        # 自顶向下的趋势混合
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []  # 输出列表
        # 将原始输入与混合后的季节性和趋势部分相加，并恢复原始序列的长度
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend  # 相加季节性和趋势分量
            if self.channel_independence:  # 如果进行通道独立性处理，应用输出交叉层
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])  # 将处理后的输出添加到输出列表中
        return out_list  # 返回最终的输出列表



# 定义模型类，继承自 nn.Module
class Model(nn.Module):
    # 初始化函数，定义类的基本参数和层
    def __init__(self, configs):
        super(Model, self).__init__()  # 调用父类的初始化方法
        self.configs = configs  # 配置参数
        self.task_name = configs.task_name  # 任务名称
        self.seq_len = configs.seq_len  # 输入序列长度
        self.label_len = configs.label_len  # 标签长度
        self.pred_len = configs.pred_len  # 预测长度
        self.down_sampling_window = configs.down_sampling_window  # 降采样窗口大小
        self.channel_independence = configs.channel_independence  # 通道独立性标志

        # 定义可分解的历史混合块
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])  # 构建多个 PastDecomposableMixing 模块

        self.preprocess = series_decomp(configs.moving_avg)  # 时间序列分解预处理
        self.enc_in = configs.enc_in  # 输入通道数
        self.use_future_temporal_feature = configs.use_future_temporal_feature  # 是否使用未来时间特征

        # 根据通道独立性设置嵌入层
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers  # 层数

        # 定义标准化层列表
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        # 针对长短期预测任务的层设置
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),  # 输入维度
                        configs.pred_len,  # 输出维度
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            # 根据通道独立性设置投影层
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)  # 输出维度为 1
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)  # 输出维度为 c_out

                # 定义输出残差层
                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                # 定义回归层
                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

        # 针对填充或异常检测任务的层设置
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)  # 输出维度为 1
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)  # 输出维度为 c_out

        # 针对分类任务的层设置
        if self.task_name == 'classification':
            self.act = F.gelu  # 激活函数
            self.dropout = nn.Dropout(configs.dropout)  # Dropout 层
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)  # 投影到分类数

    # 定义输出投影函数
    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)  # 投影层处理
        out_res = out_res.permute(0, 2, 1)  # 转置张量
        out_res = self.out_res_layers[i](out_res)  # 通过残差层处理
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)  # 回归层处理并转置回原始维度
        dec_out = dec_out + out_res  # 将结果相加
        return dec_out  # 返回最终输出

    # 定义前编码函数
    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)  # 如果通道独立性为 1，直接返回输入
        else:
            out1_list = []  # 季节性输出列表
            out2_list = []  # 趋势性输出列表
            for x in x_list:
                x_1, x_2 = self.preprocess(x)  # 进行分解
                out1_list.append(x_1)  # 季节性部分添加到列表
                out2_list.append(x_2)  # 趋势性部分添加到列表
            return (out1_list, out2_list)  # 返回分解结果

    # 定义多尺度输入处理函数
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # 根据配置的降采样方法选择对应的池化操作
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)  # 最大池化
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)  # 平均池化
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2  # 判断 PyTorch 版本来设置 padding
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)  # 卷积池化
        else:
            return x_enc, x_mark_enc  # 如果没有匹配的池化方法，返回原始输入

        # 维度变换 B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        # 初始化原始输入和时间标记
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        # 初始化采样列表
        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))  # 添加转置后的输入
        x_mark_sampling_list.append(x_mark_enc)  # 添加时间标记

        # 循环进行多层降采样
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)  # 对原始输入进行降采样
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))  # 添加降采样后的转置输入
            x_enc_ori = x_enc_sampling  # 更新原始输入为降采样后的输入

            if x_mark_enc_mark_ori is not None:  # 如果时间标记存在
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])  # 添加降采样后的时间标记
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]  # 更新时间标记

        # 最终将所有采样结果赋值回原始变量
        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc  # 返回处理后的输入和时间标记

    # 预测函数
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # 如果使用未来的时间特征
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()  # 获取输入的批次大小、时间步长、特征数
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)  # 扩展时间标记
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)  # 嵌入时间标记
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)  # 嵌入时间标记

        # 多尺度处理输入数据
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []

        # 如果存在时间标记
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()  # 获取输入的批次大小、时间步长、特征数
                x = self.normalize_layers[i](x, 'norm')  # 归一化输入
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  # 调整张量维度
                    x_mark = x_mark.repeat(N, 1, 1)  # 扩展时间标记
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()  # 获取输入的批次大小、时间步长、特征数
                x = self.normalize_layers[i](x, 'norm')  # 归一化输入
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  # 调整张量维度
                x_list.append(x)

        # 嵌入处理
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # 获取编码输出
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # 获取编码输出
                enc_out_list.append(enc_out)

        # 使用过去的数据进行解混
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)  # 调用 PDM 块

        # 使用未来的数据进行解混
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)  # 堆叠并求和解码输出
        dec_out = self.normalize_layers[0](dec_out, 'denorm')  # 反归一化解码输出
        return dec_out  # 返回预测结果

    # 未来解混函数
    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # 对齐时间维度
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec  # 添加未来时间特征
                    dec_out = self.projection_layer(dec_out)  # 投影层
                else:
                    dec_out = self.projection_layer(dec_out)  # 投影层
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()  # 调整输出维度
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # 对齐时间维度
                dec_out = self.out_projection(dec_out, i, out_res)  # 输出投影
                dec_out_list.append(dec_out)
        return dec_out_list  # 返回解码输出列表

    # 分类函数
    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)  # 多尺度处理输入
        x_list = x_enc

        # 嵌入处理
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # 获取编码输出
            enc_out_list.append(enc_out)

        # 使用多尺度交叉注意力作为过去的编码器
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)  # 调用 PDM 块

        enc_out = enc_out_list[0]  # 获取编码输出
        output = self.act(enc_out)  # 激活函数处理
        output = self.dropout(output)  # dropout 操作
        output = output * x_mark_enc.unsqueeze(-1)  # 零化填充嵌入
        output = output.reshape(output.shape[0], -1)  # 调整输出形状
        output = self.projection(output)  # 投影到类别空间
        return output  # 返回分类结果

    # 异常检测函数
    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()  # 获取输入的批次大小、时间步长、特征数
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)  # 多尺度处理输入

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc):
            x = self.normalize_layers[i](x, 'norm')  # 归一化输入
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  # 调整张量维度
            x_list.append(x)

        # 嵌入处理
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # 获取编码输出
            enc_out_list.append(enc_out)

        # 使用多尺度交叉注意力作为过去的编码器
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)  # 调用 PDM 块

        dec_out = self.projection_layer(enc_out_list[0])  # 投影层
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()  # 调整输出维度
        dec_out = self.normalize_layers[0](dec_out, 'denorm')  # 反归一化解码输出
        return dec_out  # 返回异常检测结果

    # 定义插补函数，用于处理缺失数据的插补
    def imputation(self, x_enc, x_mark_enc, mask):
        # 计算每个时间序列的均值
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)  # dim=1表示沿时间维度求和
        means = means.unsqueeze(1).detach()  # 增加一个维度以匹配输入数据的维度，并从计算图中分离

        # 数据去均值化
        x_enc = x_enc - means  # 每个时间步的数据减去均值
        x_enc = x_enc.masked_fill(mask == 0, 0)  # 对掩码为0的部分填充为0

        # 计算标准差
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)  # 标准差的计算考虑了小数值稳定性
        stdev = stdev.unsqueeze(1).detach()  # 增加一个维度并从计算图中分离

        # 数据标准化处理
        x_enc /= stdev  # 将输入数据标准化

        # 获取输入数据的批次大小(B)，时间步长(T)，特征数(N)
        B, T, N = x_enc.size()  # 获取输入张量的大小

        # 多尺度处理输入数据
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)  # 处理多尺度输入

        # 初始化列表以存储处理后的输入数据和时间标记
        x_list = []  # 用于存储处理后的输入数据
        x_mark_list = []  # 用于存储处理后的时间标记

        # 如果时间标记不为空
        if x_mark_enc is not None:
            # 对输入数据和时间标记进行遍历和处理
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()  # 获取输入的批次大小、时间步长、特征数

                # 如果通道独立性为1（表示不同通道的数据独立处理）
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  # 调整张量维度
                x_list.append(x)  # 将处理后的输入数据添加到列表中
                x_mark = x_mark.repeat(N, 1, 1)  # 扩展时间标记，使其与输入数据匹配
                x_mark_list.append(x_mark)  # 将处理后的时间标记添加到列表中

        # 如果时间标记为空
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()  # 获取输入的批次大小、时间步长、特征数

                # 如果通道独立性为1
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  # 调整张量维度
                x_list.append(x)  # 将处理后的输入数据添加到列表中

        # 嵌入层处理，将输入数据转换为高维表示
        enc_out_list = []  # 初始化列表用于存储编码输出
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # 通过嵌入层处理输入数据
            enc_out_list.append(enc_out)  # 将编码后的输出添加到列表中

        # 使用多尺度交叉注意力作为过去的编码器
        for i in range(self.layer):  # 遍历层数
            enc_out_list = self.pdm_blocks[i](enc_out_list)  # 通过PDM块处理编码输出

        # 投影层处理编码后的输出
        dec_out = self.projection_layer(enc_out_list[0])  # 获取投影层输出
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()  # 调整输出形状

        # 将标准化的数据反归一化
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))  # 乘以标准差以反标准化
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))  # 加上均值以反标准化

        return dec_out  # 返回插补后的输出


    # 前向传播函数，根据任务类型选择对应的处理方法
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 长期或短期预测任务
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # 调用预测函数
            return dec_out  # 返回预测结果

        # 数据插补任务
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)  # 调用插补函数
            return dec_out  # 返回插补结果

        # 异常检测任务
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)  # 调用异常检测函数
            return dec_out  # 返回检测结果

        # 分类任务
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)  # 调用分类函数
            return dec_out  # 返回分类结果

        # 未实现的任务类型
        else:
            raise ValueError('Other tasks not implemented yet')  # 抛出异常，提示未实现的任务类型

