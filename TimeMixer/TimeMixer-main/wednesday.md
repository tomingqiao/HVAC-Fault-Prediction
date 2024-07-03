 ## 爆改data_loader.py

```python
class Dataset_Custom(Dataset):  # 继承自 PyTorch 的 Dataset 类，用于加载自定义数据集（Brick 格式）
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
        results = graph.query(query)
        data_list = []
        def safe_float_convert(value):
            if isinstance(value, rdflib.term.Literal) and value.value == '':
                # 使用 math.nan 表示缺失值
                return math.nan 
            else:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return math.nan

        for row in results:
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
                                              'ZoneTemp_1', 'ZoneTemp_2'])

        print("Data shape:", data.shape)  # 添加这行test
        print(data.head())  # 添加这行以查看数据的前几行test

        cols = list(data.columns)
        cols.remove(self.target)
        cols.remove('timestamp')
        data = data[['timestamp'] + cols + [self.target]]
        # data['timestamp'] = pd.to_datetime(data['timestamp'], format="%Y-%m-%d_%H:%M:%S", errors='coerce')

        # data = data.dropna(subset=['timestamp'])
        # 数据分割
        # 这里假设数据按时间顺序排列，并按照 70%、20%、10% 的比例划分训练集、测试集和验证集
        # data = data.sort_values(by='timestamp')
        num_samples = len(data)
        num_train = int(num_samples * 0.7)
        num_test = int(num_samples * 0.2)
        num_val = num_samples - num_train - num_test

        border1s = [0, num_train - self.seq_len, num_samples - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, num_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = data.columns[2:]
            df_data = data[cols_data]
        else:
            df_data = data[[self.target]]

        # 标准化
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_scaled = self.scaler.transform(df_data.values)
            df_data.loc[:, :] = data_scaled
        else:
            data_scaled = df_data.values 
        # 时间编码
        self.data_stamp = pd.to_datetime(data['timestamp'], format='Y-%m-%d_%H:%M:%S', errors='coerce')
        if self.timeenc == 0:  # 使用月份、日期、星期几、小时作为时间特征
            df_stamp = pd.DataFrame({'date': self.data_stamp})
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            self.data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:  # 使用 time_features 函数生成时间特征
            self.data_stamp = pd.Index(self.data_stamp)
            self.data_stamp = time_features(self.data_stamp, freq=self.freq).transpose(1, 0)


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
        return len(self.data_x)  # 返回数据集长度

    def inverse_transform(self, data):  # 反标准化的方法
        return self.scaler.inverse_transform(data)  # 对数据进行反标准化
 
```

## 小改data_factory.py

```python
	from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_PEMS, \
    Dataset_Solar
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'PEMS': Dataset_PEMS,
    'Solar': Dataset_Solar,
}

def pad_sequences(sequences, max_len):
    padded_sequences = torch.zeros((len(sequences), max_len, sequences[0].shape[1]))
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded_sequences[i, :length] = seq
    return padded_sequences

def custom_collate_fn(batch):
    features, targets, features_mark, targets_mark = zip(*batch)

    max_seq_len_x = max([f.shape[0] for f in features])
    max_seq_len_y = max([t.shape[0] for t in targets])

    features = pad_sequences([torch.tensor(f, dtype=torch.float32).nan_to_num() for f in features], max_seq_len_x)
    targets = pad_sequences([torch.tensor(t, dtype=torch.float32).nan_to_num() for t in targets], max_seq_len_y)
    features_mark = pad_sequences([torch.tensor(fm, dtype=torch.float32).nan_to_num() for fm in features_mark], max_seq_len_x)
    targets_mark = pad_sequences([torch.tensor(tm, dtype=torch.float32).nan_to_num() for tm in targets_mark], max_seq_len_y)

    return features, targets, features_mark, targets_mark


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = args.batch_size  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            collate_fn=custom_collate_fn,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
            drop_last=drop_last
#            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            collate_fn=custom_collate_fn,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

```

## 重要发现

重要参数需要手动设置'--enc_in', '20',    '--dec_in', '20',    '--c_out', '20', '--target', '这个设置为需要预测的参数	'
```python
parameters = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', './ttl/',
    '--data_path', 'brick_model_1.3_batch_1.ttl',
    '--model_id', 'Data_Article_Dataset_96_96',
    '--model', 'TimeMixer',
    '--data', 'custom',
    '--features', 'M',
    '--target', 'OaTemp',
    '--seq_len', '96',
    '--label_len', '0',
    '--pred_len', '96',
    '--e_layers', '2',
    '--enc_in', '20',
    '--dec_in', '20',
    '--c_out', '20',
    '--des', 'Exp',
    '--itr', '1',
    '--d_model', '16',
    '--d_ff', '32',
    '--batch_size', '16',
    '--learning_rate', '0.01',
    '--down_sampling_layers', '3',
    '--down_sampling_method', 'avg',
    '--down_sampling_window', '2'
]
```
## 数据集解读
- ChWVlvPos: 冷却水阀门开度 (Chilled Water Valve Position)
- DaFanPower: 送风机功率 (Discharge Air Fan Power)
- CCALTemp: 冷却盘管出风温度 (Cooling Coil Air Leaving Temperature)
- DaTemp: 送风温度 (Discharge Air Temperature)
- EaDmprPos: 排风风阀开度 (Exhaust Air Damper Position)
- HCALTemp: 加热盘管出风温度 (Heating Coil Air Leaving Temperature)
- HWVlvPos: 热水阀门开度 (Hot Water Valve Position)
- MaTemp: 混合空气温度 (Mixed Air Temperature)
- OaDmprPos: 新风风阀开度 (Outside Air Damper Position)
- OaTemp: 新风温度 (Outside Air Temperature)
- OaTemp_WS: 气象站新风温度 (Outside Air Temperature from Weather Station)
- RaDmprPos: 回风风阀开度 (Return Air Damper Position)
- RaFanPower: 回风机功率 (Return Air Fan Power)
- RaTemp: 回风温度 (Return Air Temperature)
- ReHeatVlvPos_1: 再热阀门 1 开度 (Reheat Valve Position 1)
- ReHeatVlvPos_2: 再热阀门 2 开度 (Reheat Valve Position 2)
- ZoneDaTemp_1: 区域 1 送风温度 (Zone 1 Discharge Air Temperature)
- ZoneDaTemp_2: 区域 2 送风温度 (Zone 2 Discharge Air Temperature)
- ZoneTemp_1: 区域 1 温度 (Zone 1 Temperature)
- ZoneTemp_2: 区域 2 温度 (Zone 2 Temperature)

## ***故障预测***关键参数设置AI建议

根据您提供的 TimeMixer 参数列表和 HVAC 故障预测的目标，以下列出了一些关键参数的设置建议：
1. 任务和模型定义:
`--task_name`: long_term_forecast
由于您要预测另一台空调的故障时间，这属于长期预测范畴，而非短期预测或者其他任务。
`--model`: TimesNet
根据您的描述，您打算使用 TimesNet 模型。
`--features`: M (Multivariate predict multivariate)
您的数据集包含多个传感器数据，属于多变量时间序列。
`--c_out`: 1（有待商榷，到底是输出所有传感器数据的预测值还是输出故障发生时间的预测值？）
预测目标是故障时间，属于单变量输出。
2. 数据加载:
`--data`: 自定义数据名称，例如 'HVAC'
`--root_path`: HVAC 数据集存放的根目录
`--data_path`: HVAC 数据集文件路径，例如 './data/HVAC/data.csv'
`--freq`: 选择合适的时间频率，例如 'h' (hourly)
`--seq_len`:
根据您想使用多长的历史数据进行预测，例如，使用过去 24 小时的传感器数据预测未来 1 小时是否发生故障，则设置为 24。
`--label_len`:
通常设置为预测窗口长度的一半或更小，例如 pred_len 为 1，则可设置为 1。
`--pred_len`: 1
您需要预测未来 1 小时是否发生故障。
3. 模型定义:
`--enc_in`: 20
您的数据集包含 20 个传感器数据字段。
`--dec_in`: 根据是否使用未来时间特征决定，如不需要可与 enc_in 相同。
`--d_model`: 可以尝试 16、32、64 等值，根据模型表现进行调整.
`--n_heads`: 可以尝试 4、8 等值，通常与 d_model 成比例
`--e_layers`: 编码器层数，可以尝试 2、3 层
`--d_layers`: 解码器层数，可以尝试 1、2 层
`--dropout`: 可以尝试 0.1、0.2 等值，防止过拟合。
4. 优化:
`--batch_size`: 可以尝试 16、32、64 等值，根据您的计算资源和数据集大小进行调整。
`--train_epochs`: 可以先尝试 10、20 等 epochs，根据模型收敛情况进行调整。
`--learning_rate`: 可以尝试 0.001、0.0001 等值，通常需要根据 batch_size 进行调整。
5. 其他:
`--down_sampling_layers`, `--down_sampling_window`, `--down_sampling_method` :
这些参数与模型的下采样操作相关，需要根据您的数据特性和预测需求进行调整。如果数据没有明显的周期性，可以不进行下采样操作。
`--use_future_temporal_feature`:
如果您的预测任务中包含可获取的未来时间特征，例如节假日安排、天气预报等，可以设置为 1 并提供相应的特征数据。