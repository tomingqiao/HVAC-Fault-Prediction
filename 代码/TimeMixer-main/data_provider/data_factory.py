import torch  # 导入 PyTorch 库

# 从 data_provider.data_loader 模块导入自定义数据集类 Dataset_Custom3
from data_provider.data_loader import Dataset_Custom3  
# Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
# MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_PEMS, \
# Dataset_Solar, Dataset_Custom2,

# 从 data_provider.uea 模块导入 collate_fn 函数
from data_provider.uea import collate_fn  
from torch.utils.data import DataLoader  # 导入 PyTorch 的数据加载器模块

# 创建一个字典 data_dict，将数据集名称映射到相应的数据集类
data_dict = {
    # 'ETTh1': Dataset_ETT_hour,
    # 'ETTh2': Dataset_ETT_hour,
    # 'ETTm1': Dataset_ETT_minute,
    # 'ETTm2': Dataset_ETT_minute,
    # 'custom': Dataset_Custom,
    # 'custom2': Dataset_Custom2,
    # 'm4': Dataset_M4,
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
    # 'UEA': UEAloader,
    # 'PEMS': Dataset_PEMS,
    # 'Solar': Dataset_Solar,
    'custom3': Dataset_Custom3,  # 将字符串 'custom3' 映射到自定义数据集类 Dataset_Custom3
}

# 定义一个函数，将输入的多个序列填充到相同的长度
def pad_sequences(sequences, max_len):
    # 初始化一个全为零的张量，用于存储填充后的序列
    # 张量维度为 (序列数量, 最大长度, 每个序列的特征数)
    padded_sequences = torch.zeros((len(sequences), max_len, sequences[0].shape[1]))
    
    # 遍历每个序列，进行填充操作
    for i, seq in enumerate(sequences):
        # 获取当前序列的长度
        length = seq.shape[0]
        # 将当前序列的数据复制到填充张量的相应位置
        padded_sequences[i, :length] = seq
    
    # 返回填充后的序列张量
    return padded_sequences

# 定义一个自定义的批处理函数，对一个批次的数据进行处理
def custom_collate_fn(batch):
    # 解包 batch 中的元素，将其分别赋值给特征、目标、特征时间标记和目标时间标记
    features, targets, features_mark, targets_mark = zip(*batch)

    # 计算所有特征序列的最大长度
    max_seq_len_x = max([f.shape[0] for f in features])
    # 计算所有目标序列的最大长度
    max_seq_len_y = max([t.shape[0] for t in targets])

    # 使用 pad_sequences 函数对特征进行填充
    # 将序列中的 NaN 值替换为 0，并将数据类型转换为 float32
    features = pad_sequences([torch.tensor(f, dtype=torch.float32).nan_to_num() for f in features], max_seq_len_x)
    # 使用相同的方式对目标序列进行填充
    targets = pad_sequences([torch.tensor(t, dtype=torch.float32).nan_to_num() for t in targets], max_seq_len_y)
    # 对特征时间标记进行填充
    features_mark = pad_sequences([torch.tensor(fm, dtype=torch.float32).nan_to_num() for fm in features_mark], max_seq_len_x)
    # 对目标时间标记进行填充
    targets_mark = pad_sequences([torch.tensor(tm, dtype=torch.float32).nan_to_num() for tm in targets_mark], max_seq_len_y)

    # 返回填充后的特征、目标、特征时间标记和目标时间标记
    return features, targets, features_mark, targets_mark

# 定义一个数据提供函数，返回指定数据集的数据集和数据加载器
def data_provider(args, flag):
    # 从 data_dict 中获取指定的数据集类
    Data = data_dict[args.data]
    # 根据是否使用时间编码器决定 timeenc 的值
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':  # 如果当前数据集是测试集
        shuffle_flag = False  # 不打乱数据顺序
        drop_last = True  # 丢弃最后一个不足 batch_size 的批次
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size  # 根据任务类型设置批量大小
        else:
            batch_size = args.batch_size  # 评估时的批量大小，通常为 1
        freq = args.freq  # 设置数据频率（如 'h' 表示小时）
    else:  # 如果当前数据集是训练集或验证集
        shuffle_flag = True  # 打乱数据顺序
        drop_last = True  # 丢弃最后一个不足 batch_size 的批次
        batch_size = args.batch_size  # 训练或验证时的批量大小
        freq = args.freq  # 设置数据频率

    if args.task_name == 'anomaly_detection':  # 如果任务是异常检测
        drop_last = False  # 不丢弃最后一个批次
        # 创建异常检测数据集
        data_set = Data(
            root_path=args.root_path,  # 数据集的根目录路径
            win_size=args.seq_len,  # 窗口大小，即序列长度
            flag=flag,  # 标志变量，用于区分不同的数据加载模式或阶段
        )
        print(flag, len(data_set))  # 打印当前标志变量的值和数据集的样本数量
        # 创建数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,  # 每个批次的样本数量
            shuffle=shuffle_flag,  # 是否对数据进行随机打乱
            collate_fn=custom_collate_fn,  # 使用自定义的批处理函数
            num_workers=args.num_workers,  # 使用的线程数
            drop_last=drop_last  # 是否丢弃最后一个批次
        )
        return data_set, data_loader  # 返回数据集和数据加载器
    elif args.task_name == 'classification':  # 如果任务是分类
        drop_last = False  # 不丢弃最后一个批次
        # 创建分类数据集
        data_set = Data(
            root_path=args.root_path,  # 数据集的根目录路径
            flag=flag,  # 标志变量，用于区分不同的数据加载模式或阶段
        )
        print(flag, len(data_set))  # 打印当前标志变量的值和数据集的样本数量
        # 创建数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,  # 每个批次的样本数量
            shuffle=shuffle_flag,  # 是否对数据进行随机打乱
            collate_fn=custom_collate_fn,  # 使用自定义的批处理函数
            num_workers=args.num_workers,  # 使用的线程数
            drop_last=drop_last  # 是否丢弃最后一个批次
        )
        return data_set, data_loader  # 返回数据集和数据加载器
    else:  # 处理其他任务
        if args.data == 'm4':
            drop_last = False  # 如果数据集是 'm4'，不丢弃最后一个批次
        # 创建其他类型的数据集
        data_set = Data(
            root_path=args.root_path,  # 数据集的根目录路径
            data_path=args.data_path,  # 数据集的名称
            flag=flag,  # 标志变量，用于区分不同的数据加载模式或阶段
            size=[args.seq_len, args.label_len, args.pred_len],  # 设置序列长度、标签长度和预测长度
            features=args.features,  # 特征设置
            target=args.target,  # 目标设置
            timeenc=timeenc,  # 时间编码器设置
            freq=freq,  # 数据频率设置
            seasonal_patterns=args.seasonal_patterns  # 季节性模式设置
        )
        print(flag, len(data_set))  # 打印当前标志变量的值和数据集的样本数量
        # 创建数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,  # 每个批次的样本数量
            shuffle=shuffle_flag,  # 是否对数据进行随机打乱
            collate_fn=custom_collate_fn,  # 使用自定义的批处理函数
            num_workers=args.num_workers,  # 使用的线程数
            drop_last=drop_last  # 是否丢弃最后一个批次
        )
        return data_set, data_loader  # 返回数据集和数据加载器


# def data_provider(args, flag):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed != 'timeF' else 1
#
#     if flag == 'test':
#         shuffle_flag = False
#         drop_last = True
#         if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
#             batch_size = args.batch_size
#         else:
#             batch_size = args.batch_size  # bsz=1 for evaluation
#         freq = args.freq
#     else:
#         shuffle_flag = True
#         drop_last = True
#         batch_size = args.batch_size  # bsz for train and valid
#         freq = args.freq
#
#     if args.task_name == 'anomaly_detection':
#         drop_last = False
#         data_set = Data(
#             root_path=args.root_path,
#             win_size=args.seq_len,
#             flag=flag,
#         )
#         print(flag, len(data_set))
#         data_loader = DataLoader(
#             data_set,
#             batch_size=batch_size,
#             shuffle=shuffle_flag,
#             num_workers=args.num_workers,
#             drop_last=drop_last)
#         return data_set, data_loader
#     elif args.task_name == 'classification':
#         drop_last = False
#         data_set = Data(
#             root_path=args.root_path,
#             flag=flag,
#         )
#         print(flag, len(data_set))
#         data_loader = DataLoader(
#             data_set,
#             batch_size=batch_size,
#             shuffle=shuffle_flag,
#             num_workers=args.num_workers,
#             drop_last=drop_last,
#             collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
#         )
#         return data_set, data_loader
#     else:
#         if args.data == 'm4':
#             drop_last = False
#         data_set = Data(
#             root_path=args.root_path,
#             data_path=args.data_path,
#             flag=flag,
#             size=[args.seq_len, args.label_len, args.pred_len],
#             features=args.features,
#             target=args.target,
#             timeenc=timeenc,
#             freq=freq,
#             seasonal_patterns=args.seasonal_patterns
#         )
#         print(flag, len(data_set))
#         data_loader = DataLoader(
#             data_set,
#             batch_size=batch_size,
#             shuffle=shuffle_flag,
#             num_workers=args.num_workers,
#             drop_last=drop_last)
#         return data_set, data_loader