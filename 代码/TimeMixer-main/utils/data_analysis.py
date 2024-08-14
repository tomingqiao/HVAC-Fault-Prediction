import numpy as np
from scipy.stats import entropy

# 定义计算可预测性度量的函数
def forecastabilty(ts):
    """计算时间序列的可预测性度量。

    Args:
        ts: 输入的时间序列（numpy数组）

    Returns:
        返回一个值，表示时间序列的可预测性度量。
        计算方式是 1 - (时间序列的傅里叶变换的熵 / 白噪声的熵)
    """
    # 对时间序列进行归一化处理
    ts = (ts - ts.min())/(ts.max() - ts.min() + 0.1)
    # 计算时间序列的傅里叶变换的幅值
    fourier_ts = abs(np.fft.rfft(ts))
    # 对傅里叶变换结果进行归一化处理
    fourier_ts = (fourier_ts - fourier_ts.min()) / (
        fourier_ts.max() - fourier_ts.min())
    fourier_ts /= fourier_ts.sum()  # 归一化到总和为1
    entropy_ts = entropy(fourier_ts)  # 计算傅里叶变换后的熵
    fore_ts = 1 - entropy_ts / (np.log(len(ts)))  # 计算可预测性度量
    if np.isnan(fore_ts):  # 如果计算结果为NaN，则返回0
        return 0
    return fore_ts  # 返回可预测性度量值

# 定义计算移动窗口的可预测性度量的函数
def forecastabilty_moving(ts, window, jump=1):
    """计算移动窗口的可预测性度量。

    Args:
        ts: 输入的时间序列（numpy数组）
        window: 窗口大小，即每个子序列的长度
        jump: 移动步长，默认为1

    Returns:
        返回一个列表，包含所有子序列的可预测性度量。
    """
    # 如果时间序列长度小于等于25，则直接计算整体的可预测性度量
    if len(ts) <= 25:
        return forecastabilty(ts)
    # 计算每个子序列的可预测性度量，并存入列表
    fore_lst = np.array([
        forecastabilty(ts[i - window:i])
        for i in np.arange(window, len(ts), jump)
    ])
    # 去除计算结果中的NaN值
    fore_lst = fore_lst[~np.isnan(fore_lst)]
    return fore_lst  # 返回包含所有子序列可预测性度量的数组

# 定义趋势检测类
class Trend():
    """趋势检测类，用于检测和消除时间序列中的趋势。

    Attributes:
        ts: 输入的时间序列（numpy数组）
        train_length: 时间序列的长度
        a: 线性回归的斜率
        b: 线性回归的截距
    """

    def __init__(self, ts):
        """初始化Trend类并计算线性趋势。

        Args:
            ts: 输入的时间序列（numpy数组）
        """
        self.ts = ts
        self.train_length = len(ts)
        self.a, self.b = self.find_trend(ts)  # 计算线性趋势的斜率和截距

    # 定义计算线性趋势的函数
    def find_trend(self, insample_data):
        """找到时间序列的线性趋势。

        Args:
            insample_data: 输入的数据（numpy数组）

        Returns:
            返回线性回归的斜率和截距。
        """
        # 生成时间序列的索引作为自变量
        x = np.arange(len(insample_data))
        # 使用线性回归拟合数据，得到斜率a和截距b
        a, b = np.polyfit(x, insample_data, 1)
        return a, b  # 返回斜率和截距

    # 定义消除趋势的函数
    def detrend(self):
        """消除时间序列中的线性趋势。

        Returns:
            返回消除趋势后的时间序列。
        """
        # 使用线性回归的结果消除趋势
        return self.ts - (self.a * np.arange(0, len(self.ts), 1) + self.b)

    # 定义将趋势加回输入数据的函数
    def inverse_input(self, insample_data):
        """将线性趋势加回到输入部分的时间序列。

        Args:
            insample_data: 输入的数据（numpy数组）

        Returns:
            返回加回趋势后的时间序列。
        """
        return insample_data + (self.a * np.arange(0, len(self.ts), 1) + self.b)

    # 定义将趋势加回预测数据的函数
    def inverse_pred(self, outsample_data):
        """将线性趋势加回到预测部分的数据。

        Args:
            outsample_data: 预测的数据（numpy数组）

        Returns:
            返回加回趋势后的预测数据。
        """
        return outsample_data + (
            self.a * np.arange(self.train_length,
                               self.train_length + len(outsample_data), 1) + self.b)

# 定义季节性检测函数
def seasonality_test(original_ts, ppy):
    """检测时间序列的季节性。

    Args:
        original_ts: 输入的时间序列（numpy数组）
        ppy: 每年的周期数/频率

    Returns:
        返回一个布尔值，表示时间序列是否具有季节性。
    """
    s = acf(original_ts, 1)  # 计算滞后1的自相关系数
    for i in range(2, ppy):
        s = s + (acf(original_ts, i)**2)  # 计算所有滞后的平方和

    # 计算季节性检测的阈值
    limit = 1.645 * (np.sqrt((1 + 2 * s) / len(original_ts)))

    # 检测时间序列的ppy滞后的自相关系数是否大于阈值
    return (abs(acf(original_ts, ppy))) > limit

# 定义自相关函数
def acf(ts, k):
    """计算时间序列的自相关函数。

    Args:
        ts: 输入的时间序列（numpy数组）
        k: 滞后阶数

    Returns:
        返回指定滞后的自相关系数。
    """
    m = np.mean(ts)  # 计算时间序列的均值
    s1 = 0  # 初始化滞后的乘积和
    for i in range(k, len(ts)):
        s1 = s1 + ((ts[i] - m) * (ts[i - k] - m))  # 计算滞后的乘积和

    s2 = 0  # 初始化平方和
    for i in range(0, len(ts)):
        s2 = s2 + ((ts[i] - m)**2)  # 计算时间序列的平方和

    return float(s1 / s2)  # 返回自相关系数
