import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 库，用于数值计算
from math import sqrt  # 从数学模块导入平方根函数

from einops import rearrange, repeat  # 导入 einops 库的 rearrange 和 repeat 函数，用于张量操作

from utils.masking import TriangularCausalMask, ProbMask  # 导入自定义的掩码生成函数
from reformer_pytorch import LSHSelfAttention  # 导入 Reformer 模型的局部敏感哈希（LSH）自注意力模块


# 定义一个去平稳自注意力机制的类
class DSAttention(nn.Module):
    '''De-stationary Attention'''

    # 初始化函数
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()  # 调用父类的初始化方法
        self.scale = scale  # 缩放因子，若为 None 则默认使用 1/sqrt(E)
        self.mask_flag = mask_flag  # 掩码标志位，控制是否使用掩码
        self.output_attention = output_attention  # 控制是否输出注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # 初始化 Dropout 层，用于防止过拟合

    # 前向传播函数
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 获取输入张量的形状信息
        B, L, H, E = queries.shape  # B: 批量大小, L: 查询长度, H: 头数, E: 每个头的维度
        _, S, _, D = values.shape  # S: 键值对长度, D: 每个值的维度
        
        # 如果没有定义 scale，则使用默认的 1/sqrt(E)
        scale = self.scale or 1. / sqrt(E)

        # 如果 tau 是 None，则默认值为 1.0，形状调整为 B x 1 x 1 x 1
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        # 如果 delta 是 None，则默认值为 0.0，形状调整为 B x 1 x 1 x S
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)

        # 去平稳注意力机制，调整前软最大化分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        # 如果启用了掩码
        if self.mask_flag:
            # 如果未传入掩码，则使用三角因果掩码
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # 将掩码位置的分数设置为负无穷
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 计算注意力权重，并应用 dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 计算注意力输出
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # 根据 output_attention 的值，决定是否返回注意力权重
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# 定义一个完整的自注意力机制类
class FullAttention(nn.Module):
    # 初始化函数
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()  # 调用父类的初始化方法
        self.scale = scale  # 缩放因子，若为 None 则默认使用 1/sqrt(E)
        self.mask_flag = mask_flag  # 掩码标志位，控制是否使用掩码
        self.output_attention = output_attention  # 控制是否输出注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # 初始化 Dropout 层，用于防止过拟合

    # 前向传播函数
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 获取输入张量的形状信息
        B, L, H, E = queries.shape  # B: 批量大小, L: 查询长度, H: 头数, E: 每个头的维度
        _, S, _, D = values.shape  # S: 键值对长度, D: 每个值的维度
        
        # 如果没有定义 scale，则使用默认的 1/sqrt(E)
        scale = self.scale or 1. / sqrt(E)

        # 计算注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 如果启用了掩码
        if self.mask_flag:
            # 如果未传入掩码，则使用三角因果掩码
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # 将掩码位置的分数设置为负无穷
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 计算注意力权重，并应用 dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 计算注意力输出
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # 根据 output_attention 的值，决定是否返回注意力权重
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



# 定义一个概率自注意力机制的类
class ProbAttention(nn.Module):
    '''Probability Attention'''

    # 初始化函数，定义类的基本参数
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()  # 调用父类的初始化方法
        self.factor = factor  # 用于计算采样大小的因子
        self.scale = scale  # 缩放因子，若为 None 则默认使用 1/sqrt(D)
        self.mask_flag = mask_flag  # 掩码标志位，控制是否使用掩码
        self.output_attention = output_attention  # 控制是否输出注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # 初始化 Dropout 层，用于防止过拟合

    # 定义概率 QK 计算的私有函数
    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        '''
        Q [B, H, L, D] 表示查询张量
        K [B, H, L, D] 表示键张量
        sample_k 表示采样大小
        n_top 表示 top k 个最大值的数量
        '''
        B, H, L_K, E = K.shape  # 获取键张量的形状信息
        _, _, L_Q, _ = Q.shape  # 获取查询张量的形状信息

        # 计算采样的 QK，扩展键张量的维度以便进行采样
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # 从键张量中随机采样
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # 计算查询与采样键的点积
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # 找到稀疏性测量的 Top_k 查询
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # 使用减少后的查询张量计算 QK
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # 计算 QK 点积

        return Q_K, M_top  # 返回 QK 和 Top_k 的索引

    # 定义获取初始上下文的私有函数
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape  # 获取值张量的形状信息
        if not self.mask_flag:
            # 如果没有掩码，则使用值张量的平均值作为初始上下文
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # 如果使用掩码
            assert (L_Q == L_V)  # 确保查询和键值对的长度相等
            contex = V.cumsum(dim=-2)  # 使用累加和作为初始上下文
        return contex  # 返回初始上下文

    # 定义更新上下文的私有函数
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape  # 获取值张量的形状信息

        # 如果启用了掩码
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)  # 生成概率掩码
            scores.masked_fill_(attn_mask.mask, -np.inf)  # 掩码位置的分数设置为负无穷

        attn = torch.softmax(scores, dim=-1)  # 计算注意力权重

        # 更新上下文张量
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            # 如果输出注意力权重，则构造注意力矩阵并返回
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)  # 返回更新后的上下文和空的注意力权重

    # 定义前向传播函数
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape  # 获取查询张量的形状信息
        _, L_K, _, _ = keys.shape  # 获取键张量的形状信息

        # 转置查询、键和值张量
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # 计算采样大小 U_part 和 u
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        # 限制采样大小不超过序列长度
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # 计算 top-k 的 QK 和索引
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # 添加缩放因子
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # 获取初始上下文
        context = self._get_initial_context(values, L_Q)
        # 使用选定的 top-k 查询更新上下文
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn  # 返回上下文和注意力权重


# 定义一个注意力层的类
class AttentionLayer(nn.Module):
    '''Attention Layer'''

    # 初始化函数，定义类的基本参数
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()  # 调用父类的初始化方法

        d_keys = d_keys or (d_model // n_heads)  # 键的维度，默认为 d_model / n_heads
        d_values = d_values or (d_model // n_heads)  # 值的维度，默认为 d_model / n_heads

        self.inner_attention = attention  # 内部的注意力机制
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # 查询投影层
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)  # 键投影层
        self.value_projection = nn.Linear(d_model, d_values * n_heads)  # 值投影层
        self.out_projection = nn.Linear(d_values * n_heads, d_model)  # 输出投影层
        self.n_heads = n_heads  # 头数

    # 定义前向传播函数
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape  # 获取查询张量的形状信息
        _, S, _ = keys.shape  # 获取键张量的形状信息
        H = self.n_heads  # 头数

        # 对查询、键和值张量进行线性投影，并调整形状
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 通过内部的注意力机制计算输出和注意力权重
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)  # 调整输出张量的形状

        return self.out_projection(out), attn  # 返回最终输出和注意力权重


# 定义 Reformer 层的类，继承自 nn.Module
class ReformerLayer(nn.Module):
    # 初始化函数，定义类的基本参数
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()  # 调用父类的初始化方法
        self.bucket_size = bucket_size  # 设置桶大小，用于 LSH 注意力机制中的哈希分组
        # 初始化 LSH 自注意力机制
        self.attn = LSHSelfAttention(
            dim=d_model,  # 模型的维度
            heads=n_heads,  # 注意力头的数量
            bucket_size=bucket_size,  # 桶大小
            n_hashes=n_hashes,  # 哈希次数
            causal=causal  # 是否为因果注意力
        )

    # 定义调整序列长度的函数
    def fit_length(self, queries):
        '''
        用于确保输入序列的长度 N 是桶大小的整数倍
        '''
        B, N, C = queries.shape  # 获取查询张量的形状信息
        if N % (self.bucket_size * 2) == 0:
            return queries  # 如果长度满足条件，直接返回原查询张量
        else:
            # 如果不满足条件，则填充零值以使长度成为桶大小的整数倍
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    # 定义前向传播函数
    def forward(self, queries, keys, values, attn_mask, tau, delta):
        '''
        实现 Reformer 的前向传播过程，计算注意力输出
        '''
        B, N, C = queries.shape  # 获取查询张量的形状信息
        # 调整查询张量的长度并通过 LSH 自注意力机制进行计算
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None  # 返回查询张量和空的注意力权重


# 定义两阶段注意力层的类，继承自 nn.Module
class TwoStageAttentionLayer(nn.Module):
    '''
    两阶段注意力层 (TSA)
    输入输出形状: [batch_size, 数据维度(D), 段数(L), 模型维度(d_model)]
    '''

    # 初始化函数，定义类的基本参数
    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()  # 调用父类的初始化方法
        d_ff = d_ff or 4 * d_model  # 前馈网络隐藏层的维度，默认为 4 倍的模型维度

        # 时间注意力层，处理时间维度上的注意力机制
        self.time_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
            d_model, n_heads
        )

        # 维度发送层，处理不同维度之间的消息传递
        self.dim_sender = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
            d_model, n_heads
        )

        # 维度接收层，接收并处理维度发送层的输出
        self.dim_receiver = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
            d_model, n_heads
        )

        # 路由器，用于在维度间传递信息
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)  # Dropout 层，用于防止过拟合

        # 归一化层，用于标准化输入
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # 多层感知机 (MLP)，用于进一步处理注意力层的输出
        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 线性变换，将输入映射到更高维空间
            nn.GELU(),  # 激活函数
            nn.Linear(d_ff, d_model)  # 线性变换，将输出映射回原始维度
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 同上
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    # 定义前向传播函数
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        '''
        实现两阶段注意力机制的前向传播过程
        '''
        batch = x.shape[0]  # 获取批次大小

        # 第一个阶段：跨时间的注意力机制
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')  # 重新排列张量的维度
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)  # 对注意力输出应用 Dropout 并与输入相加
        dim_in = self.norm1(dim_in)  # 归一化
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))  # 通过 MLP 处理后再加回原输入
        dim_in = self.norm2(dim_in)  # 归一化

        # 第二个阶段：跨维度的注意力机制
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)  # 重新排列张量的维度
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)  # 扩展路由器以匹配批次大小
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)  # 维度发送操作
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)  # 维度接收操作
        dim_enc = dim_send + self.dropout(dim_receive)  # 对接收结果应用 Dropout 并与输入相加
        dim_enc = self.norm3(dim_enc)  # 归一化
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))  # 通过 MLP 处理后再加回原输入
        dim_enc = self.norm4(dim_enc)  # 归一化

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)  # 将张量的维度恢复到初始状态

        return final_out  # 返回最终的输出张量