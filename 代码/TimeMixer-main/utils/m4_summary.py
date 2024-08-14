# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright 2020 Element AI Inc. All rights reserved.

"""
M4 Summary
"""
# 导入OrderedDict模块，用于有序字典
from collections import OrderedDict

# 导入NumPy库，用于科学计算
import numpy as np
# 导入Pandas库，用于数据处理
import pandas as pd

# 从data_provider.m4模块中导入M4Dataset和M4Meta类
from data_provider.m4 import M4Dataset
from data_provider.m4 import M4Meta
# 导入os模块，用于文件和路径操作
import os

# 定义group_values函数，用于根据指定分组提取相应的非NaN值
def group_values(values, groups, group_name):
    """
    根据分组提取指定组名下的非NaN值。
    
    参数:
    - values: 需要处理的值（数组）
    - groups: 对应的分组标签
    - group_name: 要提取的组名

    返回:
    - 提取后的非NaN值数组
    """
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])

# 定义mase函数，用于计算MASE损失
def mase(forecast, insample, outsample, frequency):
    """
    计算MASE（Mean Absolute Scaled Error）损失。
    
    参数:
    - forecast: 预测值
    - insample: 样本内数据
    - outsample: 样本外数据
    - frequency: 频率参数

    返回:
    - MASE损失值
    """
    # 计算预测值与样本外数据的平均绝对误差，并除以样本内数据的平均绝对误差
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))

# 定义smape_2函数，用于计算sMAPE损失
def smape_2(forecast, target):
    """
    计算sMAPE（Symmetric Mean Absolute Percentage Error）损失。
    
    参数:
    - forecast: 预测值
    - target: 目标值

    返回:
    - sMAPE损失值
    """
    denom = np.abs(target) + np.abs(forecast)
    # 将分母为0的情况替换为1，以避免除以0
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom

# 定义mape函数，用于计算MAPE损失
def mape(forecast, target):
    """
    计算MAPE（Mean Absolute Percentage Error）损失。
    
    参数:
    - forecast: 预测值
    - target: 目标值

    返回:
    - MAPE损失值
    """
    denom = np.abs(target)
    # 将分母为0的情况替换为1，以避免除以0
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom

# 定义M4Summary类，用于加载数据集并计算模型的评估指标
class M4Summary:
    # 初始化方法，加载训练集和测试集，并设定文件路径
    def __init__(self, file_path, root_path):
        """
        初始化M4Summary类，加载数据集。
        
        参数:
        - file_path: 预测结果文件路径
        - root_path: 数据集根目录
        """
        self.file_path = file_path  # 预测结果文件路径
        self.training_set = M4Dataset.load(training=True, dataset_file=root_path)  # 加载训练集
        self.test_set = M4Dataset.load(training=False, dataset_file=root_path)  # 加载测试集
        self.naive_path = os.path.join(root_path, 'submission-Naive2.csv')  # Naive2预测文件路径

    # 定义evaluate方法，用于评估模型性能
    def evaluate(self):
        """
        使用M4测试数据集评估预测结果。
        
        :return: 按季节模式分组的sMAPE和OWA
        """
        grouped_owa = OrderedDict()  # 有序字典，用于存储按季节模式分组的OWA值

        # 加载Naive2基准预测值
        naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)
        naive2_forecasts = np.array([v[~np.isnan(v)] for v in naive2_forecasts])

        # 初始化用于存储MASE、sMAPE和MAPE的字典
        model_mases = {}
        naive2_smapes = {}
        naive2_mases = {}
        grouped_smapes = {}
        grouped_mapes = {}
        
        # 遍历所有季节模式（例如：Yearly, Quarterly等）
        for group_name in M4Meta.seasonal_patterns:
            # 根据组名构建文件路径
            file_name = self.file_path + group_name + "_forecast.csv"
            # 检查文件是否存在
            if os.path.exists(file_name):
                model_forecast = pd.read_csv(file_name).values

            # 提取该组名下的Naive2预测值和目标值
            naive2_forecast = group_values(naive2_forecasts, self.test_set.groups, group_name)
            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            # 获取该组名下的频率
            frequency = self.training_set.frequencies[self.test_set.groups == group_name][0]
            # 提取该组名下的样本内数据
            insample = group_values(self.training_set.values, self.test_set.groups, group_name)

            # 计算模型的MASE损失值
            model_mases[group_name] = np.mean([mase(forecast=model_forecast[i],
                                                    insample=insample[i],
                                                    outsample=target[i],
                                                    frequency=frequency) for i in range(len(model_forecast))])
            # 计算Naive2的MASE损失值
            naive2_mases[group_name] = np.mean([mase(forecast=naive2_forecast[i],
                                                     insample=insample[i],
                                                     outsample=target[i],
                                                     frequency=frequency) for i in range(len(model_forecast))])

            # 计算Naive2的sMAPE损失值
            naive2_smapes[group_name] = np.mean(smape_2(naive2_forecast, target))
            # 计算模型的sMAPE损失值
            grouped_smapes[group_name] = np.mean(smape_2(forecast=model_forecast, target=target))
            # 计算模型的MAPE损失值
            grouped_mapes[group_name] = np.mean(mape(forecast=model_forecast, target=target))

        # 汇总按组计算的sMAPE、MAPE和MASE
        grouped_smapes = self.summarize_groups(grouped_smapes)
        grouped_mapes = self.summarize_groups(grouped_mapes)
        grouped_model_mases = self.summarize_groups(model_mases)
        grouped_naive2_smapes = self.summarize_groups(grouped_naive2_smapes)
        grouped_naive2_mases = self.summarize_groups(grouped_naive2_mases)

        # 计算按组的OWA（Overall Weighted Average）指标
        for k in grouped_model_mases.keys():
            grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +
                              grouped_smapes[k] / grouped_naive2_smapes[k]) / 2

        # 定义round_all函数，将所有值四舍五入到小数点后三位
        def round_all(d):
            return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))

        # 返回汇总后的sMAPE、OWA、MAPE和MASE
        return round_all(grouped_smapes), round_all(grouped_owa), round_all(grouped_mapes), round_all(
            grouped_model_mases)

    # 定义summarize_groups方法，重新分组计算得分
    def summarize_groups(self, scores):
        """
        根据M4规则重新分组得分。
        
        :param scores: 按组计算的得分。
        :return: 重新分组后的得分。
        """
        scores_summary = OrderedDict()  # 有序字典，用于存储得分摘要

        # 定义group_count函数，计算某组中的数据量
        def group_count(group_name):
            return len(np.where(self.test_set.groups == group_name)[0])

        weighted_score = {}  # 存储加权得分
        
        # 计算Yearly, Quarterly, Monthly三组的加权得分
        for g in ['Yearly', 'Quarterly', 'Monthly']:
            weighted_score[g] = scores[g] * group_count(g)
            scores_summary[g] = scores[g]

        # 计算其他组（Weekly, Daily, Hourly）的加权得分
        others_score = 0
        others_count = 0
        for g in ['Weekly', 'Daily', 'Hourly']:
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        weighted_score['Others'] = others_score
        scores_summary['Others'] = others_score / others_count

        # 计算总平均得分
        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)
        scores_summary['Average'] = average

        # 返回重新分组后的得分摘要
        return scores_summary
