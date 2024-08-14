# 导入必要的类型提示工具，用于为函数定义输入和输出类型
from typing import List

# 导入NumPy库，用于数值计算和数组操作
import numpy as np

# 导入Pandas库，用于处理时间序列数据
import pandas as pd

# 从pandas的时间序列模块中导入offsets类，用于处理时间频率的偏移
from pandas.tseries import offsets

# 从pandas的时间序列频率模块中导入to_offset函数，用于将频率字符串转换为offset对象
from pandas.tseries.frequencies import to_offset


# 定义时间特征基类，所有具体时间特征类都将继承这个类
class TimeFeature:
    def __init__(self):
        pass  # 初始化方法，当前没有特别的初始化逻辑

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass  # 定义该类的调用接口，具体实现将在子类中完成

    def __repr__(self):
        # 返回类的名称，主要用于打印时的表示
        return self.__class__.__name__ + "()"


# 定义具体的时间特征类，表示每分钟的秒数，并将其编码为[-0.5, 0.5]之间的值
class SecondOfMinute(TimeFeature):
    """分钟内的秒数，编码为[-0.5, 0.5]之间的值"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 返回时间序列中每个时间点的秒数，并将其标准化到[-0.5, 0.5]范围内
        return index.second / 59.0 - 0.5


# 定义具体的时间特征类，表示每小时的分钟数，并将其编码为[-0.5, 0.5]之间的值
class MinuteOfHour(TimeFeature):
    """每小时的分钟数，编码为[-0.5, 0.5]之间的值"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 返回时间序列中每个时间点的分钟数，并将其标准化到[-0.5, 0.5]范围内
        return index.minute / 59.0 - 0.5


# 定义具体的时间特征类，表示一天中的小时数，并将其编码为[-0.5, 0.5]之间的值
class HourOfDay(TimeFeature):
    """一天中的小时数，编码为[-0.5, 0.5]之间的值"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 返回时间序列中每个时间点的小时数，并将其标准化到[-0.5, 0.5]范围内
        return index.hour / 23.0 - 0.5


# 定义具体的时间特征类，表示一周中的天数，并将其编码为[-0.5, 0.5]之间的值
class DayOfWeek(TimeFeature):
    """一周中的天数，编码为[-0.5, 0.5]之间的值"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 返回时间序列中每个时间点是星期几，并将其标准化到[-0.5, 0.5]范围内
        return index.dayofweek / 6.0 - 0.5


# 定义具体的时间特征类，表示一个月中的天数，并将其编码为[-0.5, 0.5]之间的值
class DayOfMonth(TimeFeature):
    """一个月中的天数，编码为[-0.5, 0.5]之间的值"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 返回时间序列中每个时间点在一个月中的哪一天，并将其标准化到[-0.5, 0.5]范围内
        return (index.day - 1) / 30.0 - 0.5


# 定义具体的时间特征类，表示一年中的天数，并将其编码为[-0.5, 0.5]之间的值
class DayOfYear(TimeFeature):
    """一年中的天数，编码为[-0.5, 0.5]之间的值"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 返回时间序列中每个时间点在一年中的第几天，并将其标准化到[-0.5, 0.5]范围内
        return (index.dayofyear - 1) / 365.0 - 0.5


# 定义具体的时间特征类，表示一年中的月份，并将其编码为[-0.5, 0.5]之间的值
class MonthOfYear(TimeFeature):
    """一年中的月份，编码为[-0.5, 0.5]之间的值"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 返回时间序列中每个时间点在一年中的月份，并将其标准化到[-0.5, 0.5]范围内
        return (index.month - 1) / 11.0 - 0.5


# 定义具体的时间特征类，表示一年中的第几周，并将其编码为[-0.5, 0.5]之间的值
class WeekOfYear(TimeFeature):
    """一年中的第几周，编码为[-0.5, 0.5]之间的值"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 返回时间序列中每个时间点在一年中的第几周，并将其标准化到[-0.5, 0.5]范围内
        return (index.isocalendar().week - 1) / 52.0 - 0.5


# 定义一个函数，用于根据给定的频率字符串返回相应的时间特征类列表
def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    根据给定的频率字符串，返回相应的时间特征列表。

    参数:
    ----------
    freq_str: str
        表示频率的字符串，例如 "12H", "5min", "1D" 等。

    返回:
    ----------
    List[TimeFeature]:
        时间特征类的实例列表，依据输入的频率字符串自动匹配合适的特征类。
    """

    # 创建一个字典，映射不同的时间偏移量到相应的时间特征类列表
    features_by_offsets = {
        offsets.YearEnd: [],  # 年末不返回任何时间特征
        offsets.QuarterEnd: [MonthOfYear],  # 季末返回月份特征
        offsets.MonthEnd: [MonthOfYear],  # 月末返回月份特征
        offsets.Week: [DayOfMonth, WeekOfYear],  # 每周返回月份天数和年度周数特征
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],  # 每天返回周天数、月天数和年天数特征
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # 每个工作日返回同上
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],  # 每小时返回小时、周天数、月天数和年天数特征
        offsets.Minute: [
            MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
        ],  # 每分钟返回分钟、小时、周天数、月天数和年天数特征
        offsets.Second: [
            SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
        ],  # 每秒钟返回秒、分钟、小时、周天数、月天数和年天数特征
    }

    # 使用to_offset函数将频率字符串转换为偏移量对象
    offset = to_offset(freq_str)

    # 遍历字典中的每个偏移量类型及其对应的时间特征类列表
    for offset_type, feature_classes in features_by_offsets.items():
        # 检查当前的偏移量对象是否属于某个特定的偏移量类型
        if isinstance(offset, offset_type):
            # 如果匹配，返回该偏移量类型所对应的时间特征类的实例列表
            return [cls() for cls in feature_classes]

    # 如果没有找到匹配的偏移量类型，则抛出错误，提示支持的频率字符串格式
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    # 抛出运行时错误，包含支持的频率格式提示信息
    raise RuntimeError(supported_freq_msg)


# 定义一个函数，用于获取给定日期序列的时间特征
def time_features(dates, freq='h'):
    """
    根据给定的日期序列和频率，生成相应的时间特征矩阵。

    参数:
    ----------
    dates: pd.DatetimeIndex
        日期时间索引，表示要提取时间特征的时间点。
    freq: str, 默认值为 'h'
        频率字符串，决定提取哪些时间特征。

    返回:
    ----------
    np.ndarray:
        包含所有提取时间特征的矩阵，每个时间特征作为矩阵的一列。
    """
    # 调用time_features_from_frequency_str函数，根据频率字符串获取时间特征类实例列表
    # 然后为每个时间特征类实例调用__call__方法，生成时间特征数组
    # 最后将所有特征数组进行堆叠，并返回结果矩阵
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
