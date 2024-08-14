# 导入argparse模块，用于解析命令行参数
import argparse

# 导入PyTorch库，用于深度学习
import torch

# 从自定义模块exp中导入不同任务的实验类
from exp.exp_anomaly_detection import Exp_Anomaly_Detection  # 导入异常检测实验类
from exp.exp_classification import Exp_Classification  # 导入分类实验类
from exp.exp_imputation import Exp_Imputation  # 导入数据插补实验类
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # 导入长期预测实验类
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast  # 导入短期预测实验类

# 导入random模块，用于生成随机数
import random
# 导入NumPy库，用于数值计算
import numpy as np

# 固定随机种子，确保实验的可重复性
fix_seed = 2021  # 固定的随机种子数值

# 设置Python内置随机数生成器的种子，确保随机操作的一致性
random.seed(fix_seed)

# 设置PyTorch随机数生成器的种子，确保模型训练的一致性
torch.manual_seed(fix_seed)

# 设置NumPy随机数生成器的种子，确保随机数相关操作的一致性
np.random.seed(fix_seed)

# 创建ArgumentParser对象，用于处理命令行输入
parser = argparse.ArgumentParser(description='TimeMixer')

# 基本配置
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='任务名称，选项：[long_term_forecast（长期预测）, short_term_forecast（短期预测）, imputation（数据插补）, classification（分类）, anomaly_detection（异常检测）]')
parser.add_argument('--is_training', type=int, required=True, default=1, 
                    help='状态标志，1表示训练模式，0表示测试模式')
parser.add_argument('--model_id', type=str, required=True, default='test', 
                    help='模型ID，用于标识实验的唯一标识符')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='模型名称，选项：[Autoformer, Transformer, TimesNet]，表示选择使用的模型')

# 数据加载配置
parser.add_argument('--data', type=str, required=True, default='ETTm1', 
                    help='数据集类型，如：ETTm1表示使用ETT数据集的子集')
parser.add_argument('--root_path', type=str, default='./data/ETT/', 
                    help='数据文件的根路径')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', 
                    help='具体的数据文件名称')
parser.add_argument('--features', type=str, default='M',
                    help='预测任务类型，选项：[M（多变量预测多变量）, S（单变量预测单变量）, MS（多变量预测单变量）]')
parser.add_argument('--target', type=str, default='OT', 
                    help='在S或MS任务中的目标特征')
parser.add_argument('--freq', type=str, default='h',
                    help='时间特征编码的频率，选项：[s:每秒, t:每分钟, h:每小时, d:每天, b:工作日, w:每周, m:每月]，还可以使用更详细的频率如15min或3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
                    help='模型检查点的保存位置')

# 预测任务配置
parser.add_argument('--seq_len', type=int, default=96, 
                    help='输入序列的长度，表示模型接收的时间步数')
parser.add_argument('--label_len', type=int, default=48, 
                    help='开始标记序列的长度，通常用于Transformer的解码器')
parser.add_argument('--pred_len', type=int, default=96, 
                    help='预测序列的长度，表示模型输出的时间步数')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', 
                    help='季节性模式，仅用于M4数据集的子集选择')
parser.add_argument('--inverse', action='store_true', 
                    help='是否反转输出数据，用于特定任务，如将预测结果逆归一化', default=False)

# 模型定义配置
parser.add_argument('--top_k', type=int, default=5, 
                    help='TimesBlock使用的Top-k值，用于选择重要特征')
parser.add_argument('--num_kernels', type=int, default=6, 
                    help='Inception模块中的卷积核数量')
parser.add_argument('--enc_in', type=int, default=7, 
                    help='编码器的输入大小，即输入特征维度')
parser.add_argument('--dec_in', type=int, default=7, 
                    help='解码器的输入大小')
parser.add_argument('--c_out', type=int, default=7, 
                    help='模型输出的维度，即输出特征维度')
parser.add_argument('--d_model', type=int, default=16, 
                    help='模型的隐藏层维度，影响模型的复杂度和表示能力')
parser.add_argument('--n_heads', type=int, default=4, 
                    help='多头注意力机制的头数，更多的头可以捕获更多样的注意力模式')
parser.add_argument('--e_layers', type=int, default=2, 
                    help='编码器的层数')
parser.add_argument('--d_layers', type=int, default=1, 
                    help='解码器的层数')
parser.add_argument('--d_ff', type=int, default=32, 
                    help='前馈神经网络的维度，通常为d_model的4倍')
parser.add_argument('--moving_avg', type=int, default=25, 
                    help='移动平均的窗口大小，用于时间序列去噪')
parser.add_argument('--factor', type=int, default=1, 
                    help='注意力机制中的缩放因子，通常用于调节注意力得分的尺度')
parser.add_argument('--distil', action='store_false',
                    help='是否在编码器中使用蒸馏，启用时会减少模型的复杂度',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, 
                    help='Dropout率，用于防止过拟合')
parser.add_argument('--embed', type=str, default='timeF',
                    help='时间特征的编码方式，选项：[timeF（时间频率编码）, fixed（固定编码）, learned（学习编码）]')
parser.add_argument('--activation', type=str, default='gelu', 
                    help='激活函数类型，常见选项有gelu、relu等')
parser.add_argument('--output_attention', action='store_true', 
                    help='是否在编码器中输出注意力权重，用于解释模型')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='是否启用通道独立性，0表示依赖，1表示独立，常用于FreTS模型')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='时间序列分解的方法，仅支持moving_avg（移动平均）或dft_decomp（傅里叶变换分解）')
parser.add_argument('--use_norm', type=int, default=1, 
                    help='是否使用归一化，1表示启用，0表示禁用')
parser.add_argument('--down_sampling_layers', type=int, default=0, 
                    help='下采样层的数量')
parser.add_argument('--down_sampling_window', type=int, default=1, 
                    help='下采样的窗口大小')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='下采样的方法，仅支持avg（平均值）、max（最大值）、conv（卷积）')
parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='是否使用未来的时间特征，1表示启用，0表示禁用')

# 数据插补任务配置
parser.add_argument('--mask_rate', type=float, default=0.25, 
                    help='掩码比例，表示数据插补任务中掩盖的比例')

# 异常检测任务配置
parser.add_argument('--anomaly_ratio', type=float, default=0.25, 
                    help='异常比例的先验信息，表示异常数据在总数据中的比例')

# 优化配置
parser.add_argument('--num_workers', type=int, default=10, 
                    help='数据加载时的线程数量')
parser.add_argument('--itr', type=int, default=1, 
                    help='实验次数，表示重复实验的次数')
parser.add_argument('--train_epochs', type=int, default=10, 
                    help='训练的轮数')
parser.add_argument('--batch_size', type=int, default=16, 
                    help='训练数据的批量大小')
parser.add_argument('--patience', type=int, default=10, 
                    help='早停法的耐心值，表示在验证集损失不再下降的情况下，最多允许多少个epoch不停止训练')
parser.add_argument('--learning_rate', type=float, default=0.001, 
                    help='优化器的学习率')
parser.add_argument('--des', type=str, default='test', 
                    help='实验描述')
parser.add_argument('--loss', type=str, default='MSE', 
                    help='损失函数类型，常见选项有MSE（均方误差）、MAE（平均绝对误差）等')
parser.add_argument('--lradj', type=str, default='TST', 
                    help='学习率调整方式')
parser.add_argument('--pct_start', type=float, default=0.2, 
                    help='学习率在调整中的起始比例')
parser.add_argument('--use_amp', action='store_true', 
                    help='是否使用自动混合精度训练，以减少显存使用并加速训练', default=False)
parser.add_argument('--comment', type=str, default='none', 
                    help='备注信息')

# GPU配置
parser.add_argument('--use_gpu', type=bool, default=True, 
                    help='是否使用GPU进行训练，1表示使用，0表示不使用')
parser.add_argument('--gpu', type=int, default=0, 
                    help='使用的GPU设备编号')
parser.add_argument('--use_multi_gpu', action='store_true', 
                    help='是否使用多GPU进行训练', default=False)
parser.add_argument('--devices', type=str, default='0,1', 
                    help='多GPU训练时使用的设备ID列表')

# 非平稳投影器参数
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='投影器中隐藏层的维度列表')
parser.add_argument('--p_hidden_layers', type=int, default=2, 
                    help='投影器中隐藏层的层数')


# 解析命令行参数，存储在args中
args = parser.parse_args()

# 判断是否使用GPU：如果GPU可用且命令行参数指定使用GPU，则设置args.use_gpu为True，否则为False
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# 如果使用GPU且指定了多GPU模式，则进行多GPU的设置
if args.use_gpu and args.use_multi_gpu:
    # 移除设备列表中的空格
    args.devices = args.devices.replace(' ', '')
    
    # 将设备ID字符串按逗号分割成列表
    device_ids = args.devices.split(',')
    
    # 将设备ID列表转换为整数列表
    args.device_ids = [int(id_) for id_ in device_ids]
    
    # 设置使用的主要GPU设备ID
    args.gpu = args.device_ids[0]

# 打印实验中使用的参数设置
print('Args in experiment:')
print(args)

# 根据任务名称选择对应的实验类
if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast  # 长期预测实验类
elif args.task_name == 'short_term_forecast':
    Exp = Exp_Short_Term_Forecast  # 短期预测实验类
elif args.task_name == 'imputation':
    Exp = Exp_Imputation  # 数据插补实验类
elif args.task_name == 'anomaly_detection':
    Exp = Exp_Anomaly_Detection  # 异常检测实验类
elif args.task_name == 'classification':
    Exp = Exp_Classification  # 分类实验类
else:
    Exp = Exp_Long_Term_Forecast  # 默认选择长期预测实验类

# 判断是否处于训练模式
if args.is_training:
    # 进行多次实验迭代
    for ii in range(args.itr):
        # 设置实验的记录标识
        setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,    # 任务名称
            args.model_id,     # 模型ID
            args.comment,      # 备注
            args.model,        # 模型名称
            args.data,         # 数据集名称
            args.seq_len,      # 序列长度
            args.pred_len,     # 预测长度
            args.d_model,      # 模型维度
            args.n_heads,      # 注意力头数
            args.e_layers,     # 编码层数
            args.d_layers,     # 解码层数
            args.d_ff,         # 前馈神经网络维度
            args.factor,       # 缩放因子
            args.embed,        # 嵌入方式
            args.distil,       # 是否使用蒸馏
            args.des,          # 描述
            ii)              # 当前迭代次数
        
        # 实例化实验类
        exp = Exp(args)
        
        # 打印开始训练的信息
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
        # 调用实验类的train方法进行训练
        exp.train(setting)

        # 打印测试的信息
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        
        # 调用实验类的test方法进行测试
        exp.test(setting)
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
else:
    # 如果不是训练模式，则直接进行一次实验测试
    ii = 0
    # 设置实验的记录标识
    setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,    # 任务名称
        args.model_id,     # 模型ID
        args.comment,      # 备注
        args.model,        # 模型名称
        args.data,         # 数据集名称
        args.seq_len,      # 序列长度
        args.pred_len,     # 预测长度
        args.d_model,      # 模型维度
        args.n_heads,      # 注意力头数
        args.e_layers,     # 编码层数
        args.d_layers,     # 解码层数
        args.d_ff,         # 前馈神经网络维度
        args.factor,       # 缩放因子
        args.embed,        # 嵌入方式
        args.distil,       # 是否使用蒸馏
        args.des,          # 描述
        ii)                # 当前迭代次数

    # 实例化实验类
    exp = Exp(args)
    
    # 打印测试的信息
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    
    # 调用实验类的test方法进行测试，test参数设置为1
    exp.test(setting, test=1)
    
    # 清空GPU缓存
    torch.cuda.empty_cache()

