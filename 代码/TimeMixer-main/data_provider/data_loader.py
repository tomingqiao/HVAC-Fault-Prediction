import math                          # 导入数学模块，用于执行数学函数和运算
import os                            # 导入操作系统模块，用于与操作系统进行交互，如文件路径处理

import numpy as np                   # 导入NumPy库，用于进行高效的数值计算和数组操作
import pandas as pd                  # 导入Pandas库，用于数据操作和分析，尤其是表格数据的处理
import glob                          # 导入glob模块，用于查找符合特定模式的文件路径
import re                            # 导入正则表达式模块，用于字符串的模式匹配和处理

import rdflib                        # 导入RDFLib库，用于处理RDF数据和语义网资源
import torch                         # 导入PyTorch库，用于深度学习和张量操作
from sktime.datasets import load_from_tsfile_to_dataframe  # 从sktime库中导入特定函数，用于将时间序列文件加载为DataFrame格式
from torch.utils.data import Dataset  # 从PyTorch导入Dataset类，用于自定义数据集
from sklearn.preprocessing import StandardScaler  # 从sklearn导入标准化缩放器，用于对特征数据进行标准化
from utils.timefeatures import time_features  # 从自定义的utils模块中导入时间特征提取函数
from data_provider.m4 import M4Dataset, M4Meta  # 从data_provider模块中导入M4数据集和元数据类
from data_provider.uea import Normalizer, interpolate_missing, subsample  # 从data_provider模块中导入归一化器、插值函数和下采样函数
import warnings                       # 导入警告模块，用于控制警告的显示

warnings.filterwarnings('ignore')     # 忽略所有警告，通常用于在不希望警告打断程序时使用

class Dataset_Custom3(Dataset): #处理brick数据的类
    def __init__(self, root_path, flag='train', size=None,  # 初始化方法，接收数据集路径、数据分割标志、数据大小等参数
                 features='S', data_path='brick_data.ttl',  # 特征类型、数据文件路径
                 target='target', scale=True, timeenc=0, freq='h', seasonal_patterns=None):  # 是否标准化、时间编码类型、频率、季节性模式
        # size [seq_len, label_len, pred_len]  # 数据大小，包括序列长度、标签长度、预测长度
        # info  # 信息
        if size == None:  # 如果没有指定数据大小，则使用默认值
            self.seq_len = 24 * 4 * 4  # 序列长度：4 天，每小时一个数据点
            self.label_len = 24 * 4  # 标签长度：1 天，每小时一个数据点
            self.pred_len = 24 * 4  # 预测长度：1 天，每小时一个数据点
        else:  # 如果指定了数据大小，则使用指定的值
            self.seq_len = size[0]  # 序列长度
            self.label_len = size[1]  # 标签长度
            self.pred_len = size[2]  # 预测长度

        # init  # 初始化
        assert flag in ['train', 'test', 'val']  # 确保数据分割标志是 'train'、'test' 或 'val' 之一
        type_map = {'train': 0, 'test': 1, 'val': 2}  # 将数据分割标志映射到数字
        self.set_type = type_map[flag]  # 获取当前数据分割类型

        self.features = features  # 特征类型
        self.target = target  # 目标特征名称
        self.scale = scale  # 是否标准化
        self.timeenc = timeenc  # 时间编码类型
        self.freq = freq  # 频率
        self.batch_size = 16
        self.root_path = root_path  # 数据集路径
        self.data_path = data_path  # 数据文件路径
        self.__read_data__()  # 读取数据

    def __read_data__(self):  # 读取数据的方法
        self.scaler = StandardScaler()  # 创建标准化器

        # 创建一个 RDF 图
        graph = rdflib.Graph()

        # 解析 TTL 文件
        graph.parse(os.path.join(self.root_path, self.data_path), format="ttl")

        # 使用 SPARQL 查询语句提取时间序列数据
        # 示例：查询所有 brick:Point 实例及其关联的时间戳和值
        query = """
        SELECT ?measurement ?timestamp ?CCALTemp ?ChWVlvPos ?DaFanPower ?DaTemp ?EaDmprPos ?HCALTemp ?HWVlvPos ?MaTemp ?OaDmprPos ?OaTemp ?OaTemp_WS ?RaDmprPos ?RaFanPower ?RaTemp ?ReHeatVlvPos_1 ?ReHeatVlvPos_2 ?ZoneDaTemp_1 ?ZoneDaTemp_2 ?ZoneTemp_1 ?ZoneTemp_2
        WHERE {
          ?measurement a brick1:Measurement ;
            rdfs:label ?timestamp ;
            brick1:CCALTemp ?CCALTemp ;
            brick1:ChWVlvPos ?ChWVlvPos ;
            brick1:DaFanPower ?DaFanPower ;
            brick1:DaTemp ?DaTemp ;
            brick1:EaDmprPos ?EaDmprPos ;
            brick1:HCALTemp ?HCALTemp ;
            brick1:HWVlvPos ?HWVlvPos ;
            brick1:MaTemp ?MaTemp ;
            brick1:OaDmprPos ?OaDmprPos ;
            brick1:OaTemp ?OaTemp ;
            brick1:OaTemp_WS ?OaTemp_WS ;
            brick1:RaDmprPos ?RaDmprPos ;
            brick1:RaFanPower ?RaFanPower ;
            brick1:RaTemp ?RaTemp ;
            brick1:ReHeatVlvPos_1 ?ReHeatVlvPos_1 ;
            brick1:ReHeatVlvPos_2 ?ReHeatVlvPos_2 ;
            brick1:ZoneDaTemp_1 ?ZoneDaTemp_1 ;
            brick1:ZoneDaTemp_2 ?ZoneDaTemp_2 ;
            brick1:ZoneTemp_1 ?ZoneTemp_1 ;
            brick1:ZoneTemp_2 ?ZoneTemp_2 .
        }
        """
        results = graph.query(query) # 执行查询并将结果赋值给变量 'results'
        data_list = [] # 创建一个空的列表用于存储数据

        def safe_float_convert(value): #安全转换数据类型为float
            if isinstance(value, rdflib.term.Literal) and value.value == '': # 如果值是 rdflib.term.Literal 对象且值为空字符串
                # 使用 math.nan 表示缺失值
                return math.nan
            else:
                try:
                    return float(value) # 尝试将值转换为浮点数
                except (TypeError, ValueError): 
                    return math.nan # 如果转换失败，返回 math.nan 以表示缺失值

        for row in results: # 遍历数据添加到data_list
            data_list.append([str(row.measurement), str(row.timestamp),
                              safe_float_convert(row.CCALTemp), safe_float_convert(row.ChWVlvPos),
                              safe_float_convert(row.DaFanPower), safe_float_convert(row.DaTemp),
                              safe_float_convert(row.EaDmprPos), safe_float_convert(row.HCALTemp),
                              safe_float_convert(row.HWVlvPos), safe_float_convert(row.MaTemp),
                              safe_float_convert(row.OaDmprPos), safe_float_convert(row.OaTemp),
                              safe_float_convert(row.OaTemp_WS), safe_float_convert(row.RaDmprPos),
                              safe_float_convert(row.RaFanPower), safe_float_convert(row.RaTemp),
                              safe_float_convert(row.ReHeatVlvPos_1), safe_float_convert(row.ReHeatVlvPos_2),
                              safe_float_convert(row.ZoneDaTemp_1), safe_float_convert(row.ZoneDaTemp_2),
                              safe_float_convert(row.ZoneTemp_1), safe_float_convert(row.ZoneTemp_2)])

        data = pd.DataFrame(data_list, columns=['measurement', 'timestamp',
                                                'CCALTemp', 'ChWVlvPos',
                                                'DaFanPower', 'DaTemp',
                                                'EaDmprPos', 'HCALTemp',
                                                'HWVlvPos', 'MaTemp',
                                                'OaDmprPos', 'OaTemp',
                                                'OaTemp_WS', 'RaDmprPos',
                                                'RaFanPower', 'RaTemp',
                                                'ReHeatVlvPos_1', 'ReHeatVlvPos_2',
                                                'ZoneDaTemp_1', 'ZoneDaTemp_2',
                                                'ZoneTemp_1', 'ZoneTemp_2']) # 为数据列表添加添加列名

        print("Data shape:", data.shape)  # 添加这行test
        print(data.head())  # 添加这行以查看数据的前几行test

        cols = list(data.columns) # 得到数据列表列名
        cols.remove(self.target) # 删除目标特征列
        cols.remove('timestamp') # 删除时间戳列
        data = data[['timestamp'] + cols + [self.target]] # 调整列的顺序
        # data['timestamp'] = pd.to_datetime(data['timestamp'], format="%Y-%m-%d_%H:%M:%S", errors='coerce')

        # data = data.dropna(subset=['timestamp'])
        # 数据分割
        # 这里假设数据按时间顺序排列，并按照 70%、20%、10% 的比例划分训练集、测试集和验证集
        data = data.sort_values(by='timestamp') # 将数据按 'timestamp' 列进行排序
        num_samples = len(data) # 获取数据样本的总数
        num_train = int(num_samples * 0.6) # 计算训练集的样本数量，为总样本数的 60%
        num_test = int(num_samples * 0.2) # 计算测试集的样本数量，为总样本数的 20%
        num_val = num_samples - num_train - num_test # 计算验证集的样本数量，为剩余的样本数

        border1s = [0, num_train - self.seq_len, num_samples - num_test - self.seq_len] # 定义各数据集的起始边界
        border2s = [num_train, num_train + num_val, num_samples] # 定义各数据集的结束边界
        border1 = border1s[self.set_type] # 获取当前集合类型对应的起始边界
        border2 = border2s[self.set_type] # 获取当前集合类型对应的结束边界

        # num_train = int(len(df_raw) * 0.7)
        # num_test = int(len(df_raw) * 0.2)
        # num_vali = len(df_raw) - num_train - num_test
        # border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali, len(df_raw)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS': # 如果特征选择为 'M' 或 'MS'
            cols_data = data.columns[2:] # 选择从第 3 列开始的所有列（即除时间戳和目标列外的所有特征列）
            df_data = data[cols_data] # 从数据集中提取这些特征列的数据
        else:
            df_data = data[[self.target]]  # 否则，只选择目标列的数据

        # 标准化
        if self.scale:
            # 提取训练集的数据进行缩放器的拟合
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_scaled = self.scaler.transform(df_data.values) # 对所有数据进行缩放
            df_data.loc[:, :] = data_scaled # 将缩放后的数据替换回原数据框
        else:
            data_scaled = df_data.values # 如果不进行缩放，直接使用原始数据
            
        # 时间编码
        self.data_stamp = pd.to_datetime(data['timestamp'], format='Y-%m-%d_%H:%M:%S', errors='coerce')
        if self.timeenc == 0:  # 使用月份、日期、星期几、小时作为时间特征
            df_stamp = pd.DataFrame({'date': self.data_stamp})
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            self.data_stamp = df_stamp.drop(['date'], axis=1).values # 去掉 'date' 列，保留其他时间特征列
        elif self.timeenc == 1:  # 使用 time_features 函数生成时间特征
            self.data_stamp = pd.Index(self.data_stamp)
            self.data_stamp = time_features(self.data_stamp, freq=self.freq).transpose(1, 0)

        # 根据边界提取所需的数据范围
        self.data_x = data_scaled[border1:border2]
        self.data_y = data_scaled[border1:border2]
        self.data_stamp = self.data_stamp[border1:border2]

    def __getitem__(self, index):  # 获取单个样本的方法
        # 直接返回对应index的batch数据
        s_begin = index  # 获取序列起始位置
        s_end = s_begin + self.seq_len  # 获取序列结束位置
        r_begin = s_end - self.label_len  # 获取标签起始位置
        r_end = r_begin + self.label_len + self.pred_len  # 获取标签结束位置

        seq_x = self.data_x[s_begin:s_end]  # 获取输入序列
        seq_y = self.data_y[r_begin:r_end]  # 获取输出序列
        seq_x_mark = self.data_stamp[s_begin:s_end]  # 获取输入序列的时间戳
        seq_y_mark = self.data_stamp[r_begin:r_end]  # 获取输出序列的时间戳

        return seq_x, seq_y, seq_x_mark, seq_y_mark  # 返回输入序列、输出序列、输入时间戳、输出时间戳

    def __len__(self):  # 获取数据集长度的方法
        return len(self.data_x) - self.seq_len - self.pred_len + 1  # 返回数据集长度

    def inverse_transform(self, data):  # 反标准化的方法
        return self.scaler.inverse_transform(data)  # 对数据进行反标准化

# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
#         border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_ETT_minute(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTm1.csv',
#                  target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
#         border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_Custom2(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path), na_values=[''])

#         df_raw = df_raw.fillna(0)
#         # for col in df_raw.columns:
#         #     if pd.api.types.is_numeric_dtype(df_raw[col]):
#         #         # 数值列，填充 NaN 值为均值
#         #         df_raw[col].fillna(df_raw[col].mean(), inplace=True)

#         # df_raw = pd.read_csv(os.path.join(self.root_path,
#         #                                   self.data_path))
#         # df_raw.interpolate(method='linear', inplace=True)
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             data_stamp = df_stamp.drop(['date'], axis=1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)





# class Dataset_M4(Dataset):
#     def __init__(self, root_path, flag='pred', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
#                  seasonal_patterns='Yearly'):
#         # size [seq_len, label_len, pred_len]
#         # init
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.root_path = root_path

#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]

#         self.seasonal_patterns = seasonal_patterns
#         self.history_size = M4Meta.history_size[seasonal_patterns]
#         self.window_sampling_limit = int(self.history_size * self.pred_len)
#         self.flag = flag

#         self.__read_data__()

#     def __read_data__(self):
#         # M4Dataset.initialize()
#         if self.flag == 'train':
#             dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
#         else:
#             dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
#         training_values = np.array(
#             [v[~np.isnan(v)] for v in
#              dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
#         self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
#         self.timeseries = [ts for ts in training_values]

#     def __getitem__(self, index):
#         insample = np.zeros((self.seq_len, 1))
#         insample_mask = np.zeros((self.seq_len, 1))
#         outsample = np.zeros((self.pred_len + self.label_len, 1))
#         outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

#         sampled_timeseries = self.timeseries[index]
#         cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
#                                       high=len(sampled_timeseries),
#                                       size=1)[0]

#         insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
#         insample[-len(insample_window):, 0] = insample_window
#         insample_mask[-len(insample_window):, 0] = 1.0
#         outsample_window = sampled_timeseries[
#                            cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
#         outsample[:len(outsample_window), 0] = outsample_window
#         outsample_mask[:len(outsample_window), 0] = 1.0
#         return insample, outsample, insample_mask, outsample_mask

#     def __len__(self):
#         return len(self.timeseries)

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

#     def last_insample_window(self):
#         """
#         The last window of insample size of all timeseries.
#         This function does not support batching and does not reshuffle timeseries.

#         :return: Last insample window of all timeseries. Shape "timeseries, insample size"
#         """
#         insample = np.zeros((len(self.timeseries), self.seq_len))
#         insample_mask = np.zeros((len(self.timeseries), self.seq_len))
#         for i, ts in enumerate(self.timeseries):
#             ts_last_window = ts[-self.seq_len:]
#             insample[i, -len(ts):] = ts_last_window
#             insample_mask[i, -len(ts):] = 1.0
#         return insample, insample_mask


# class PSMSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = pd.read_csv(os.path.join(root_path, 'train.csv'))
#         data = data.values[:, 1:]
#         data = np.nan_to_num(data)
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#         test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
#         test_data = test_data.values[:, 1:]
#         test_data = np.nan_to_num(test_data)
#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         self.val = self.test
#         self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class MSLSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = np.load(os.path.join(root_path, "MSL_train.npy"))
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#         test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         self.val = self.test
#         self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class SMAPSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = np.load(os.path.join(root_path, "SMAP_train.npy"))
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#         test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         self.val = self.test
#         self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):

#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class SMDSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=100, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = np.load(os.path.join(root_path, "SMD_train.npy"))
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#         test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         data_len = len(self.train)
#         self.val = self.train[(int)(data_len * 0.8):]
#         self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class SWATSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()

#         train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
#         test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
#         labels = test_data.values[:, -1:]
#         train_data = train_data.values[:, :-1]
#         test_data = test_data.values[:, :-1]

#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data)
#         test_data = self.scaler.transform(test_data)
#         self.train = train_data
#         self.test = test_data
#         self.val = test_data
#         self.test_labels = labels
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):
#         """
#         Number of images in the object dataset.
#         """
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class UEAloader(Dataset):
#     """
#     Dataset class for dataset included in:
#         Time Series Classification Archive (www.timeseriesclassification.com)
#     Argument:
#         limit_size: float in (0, 1) for debug
#     Attributes:
#         all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
#             Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
#         feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
#         feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
#         all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
#         labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
#         max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
#             (Moreover, script argument overrides this attribute)
#     """

#     def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
#         self.root_path = root_path
#         self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
#         self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

#         if limit_size is not None:
#             if limit_size > 1:
#                 limit_size = int(limit_size)
#             else:  # interpret as proportion if in (0, 1]
#                 limit_size = int(limit_size * len(self.all_IDs))
#             self.all_IDs = self.all_IDs[:limit_size]
#             self.all_df = self.all_df.loc[self.all_IDs]

#         # use all features
#         self.feature_names = self.all_df.columns
#         self.feature_df = self.all_df

#         # pre_process
#         normalizer = Normalizer()
#         self.feature_df = normalizer.normalize(self.feature_df)
#         print(len(self.all_IDs))

#     def load_all(self, root_path, file_list=None, flag=None):
#         """
#         Loads dataset from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
#         Args:
#             root_path: directory containing all individual .csv files
#             file_list: optionally, provide a list of file paths within `root_path` to consider.
#                 Otherwise, entire `root_path` contents will be used.
#         Returns:
#             all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
#             labels_df: dataframe containing label(s) for each sample
#         """
#         # Select paths for training and evaluation
#         if file_list is None:
#             data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
#         else:
#             data_paths = [os.path.join(root_path, p) for p in file_list]
#         if len(data_paths) == 0:
#             raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
#         if flag is not None:
#             data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
#         input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
#         if len(input_paths) == 0:
#             raise Exception("No .ts files found using pattern: '{}'".format(pattern))

#         all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

#         return all_df, labels_df

#     def load_single(self, filepath):
#         df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
#                                                    replace_missing_vals_with='NaN')
#         labels = pd.Series(labels, dtype="category")
#         self.class_names = labels.cat.categories
#         labels_df = pd.DataFrame(labels.cat.codes,
#                                  dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

#         lengths = df.applymap(
#             lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

#         horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

#         if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
#             df = df.applymap(subsample)

#         lengths = df.applymap(lambda x: len(x)).values
#         vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
#         if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
#             self.max_seq_len = int(np.max(lengths[:, 0]))
#         else:
#             self.max_seq_len = lengths[0, 0]

#         # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
#         # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
#         # sample index (i.e. the same scheme as all datasets in this project)

#         df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
#             pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

#         # Replace NaN values
#         grp = df.groupby(by=df.index)
#         df = grp.transform(interpolate_missing)

#         return df, labels_df

#     def instance_norm(self, case):
#         if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
#             mean = case.mean(0, keepdim=True)
#             case = case - mean
#             stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             case /= stdev
#             return case
#         else:
#             return case

#     def __getitem__(self, ind):
#         return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
#                torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

#     def __len__(self):
#         return len(self.all_IDs)


# class Dataset_PEMS(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         # info

#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         data_file = os.path.join(self.root_path, self.data_path)
#         print('data file:', data_file)
#         data = np.load(data_file, allow_pickle=True)
#         data = data['data'][:, :, 0]

#         train_ratio = 0.6
#         valid_ratio = 0.2
#         train_data = data[:int(train_ratio * len(data))]
#         valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
#         test_data = data[int((train_ratio + valid_ratio) * len(data)):]
#         total_data = [train_data, valid_data, test_data]
#         data = total_data[self.set_type]

#         if self.scale:
#             self.scaler.fit(data)
#             data = self.scaler.transform(data)

#         df = pd.DataFrame(data)
#         df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

#         self.data_x = df
#         self.data_y = df

#     def __getitem__(self, index):
#         if self.set_type == 2:  # test:首尾相连
#             s_begin = index * 12
#         else:
#             s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = torch.zeros((seq_x.shape[0], 1))
#         seq_y_mark = torch.zeros((seq_y.shape[0], 1))

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         if self.set_type == 2:  # test:首尾相连
#             return (len(self.data_x) - self.seq_len - self.pred_len + 1) // 12
#         else:
#             return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_Solar(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = []
#         with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
#             for line in f.readlines():
#                 line = line.strip('\n').split(',')  # 去除文本中的换行符
#                 data_line = np.stack([float(i) for i in line])
#                 df_raw.append(data_line)
#         df_raw = np.stack(df_raw, 0)
#         df_raw = pd.DataFrame(df_raw)
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         df_data = df_raw.values

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data)
#             data = self.scaler.transform(df_data)
#         else:
#             data = df_data

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = torch.zeros((seq_x.shape[0], 1))
#         seq_y_mark = torch.zeros((seq_y.shape[0], 1))

#         return seq_x, seq_y, seq_x_mark, seq_x_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
