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
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4 Dataset
"""
from dataclasses import dataclass  # 从标准库中导入 dataclass 装饰器，用于简化数据类的定义

import numpy as np                # 导入 NumPy 库，用于数值计算和数组操作
import pandas as pd               # 导入 Pandas 库，用于数据操作和分析，特别是表格数据的处理
import logging                    # 导入日志记录模块，用于记录运行时的信息、警告和错误
import os                         # 导入操作系统模块，用于与操作系统进行交互，如文件路径处理
import pathlib                    # 导入 pathlib 模块，用于面向对象的文件和目录操作
import sys                        # 导入系统模块，提供与 Python 解释器交互的函数和变量
from urllib import request        # 从 urllib 库中导入 request 模块，用于处理 URL 请求，如下载文件



def url_file_name(url: str) -> str:
    """
    从 URL 中提取文件名。

    :param url: 要从中提取文件名的 URL。
    :return: 文件名。
    """
    return url.split('/')[-1] if len(url) > 0 else ''  # 使用 '/' 分割 URL 并返回最后一部分作为文件名，如果 URL 为空则返回空字符串


def download(url: str, file_path: str) -> None:
    """
    下载文件到指定路径。

    :param url: 要下载的 URL。
    :param file_path: 下载内容保存的路径。
    """

    def progress(count, block_size, total_size):
        # 计算下载进度并打印到控制台
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        # 打印下载进度到控制台
        # '\r' 用于回到行首，覆盖之前的输出
        sys.stdout.write('\rDownloading {} to {} {:.1f}%'.format(url, file_path, progress_pct))
        sys.stdout.flush() # 刷新输出缓冲区，确保进度条即时显示

    if not os.path.isfile(file_path):  # 检查文件是否已经存在
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]  # 添加用户代理信息以模拟浏览器请求
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)  # 创建保存文件的目录
        f, _ = request.urlretrieve(url, file_path, progress)  # 下载文件并显示进度
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)  # 获取文件信息
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')  # 记录下载成功的日志
    else:
        file_info = os.stat(file_path)  # 如果文件已经存在，获取其信息
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')  # 记录文件已存在的日志


@dataclass()
class M4Dataset:
    ids: np.ndarray  # 存储数据集的 ID
    groups: np.ndarray  # 存储数据集的分组信息
    frequencies: np.ndarray  # 存储数据集的频率信息
    horizons: np.ndarray  # 存储预测的范围
    values: np.ndarray  # 存储实际数据值

    @staticmethod
    def load(training: bool = True, dataset_file: str = '../dataset/m4') -> 'M4Dataset':
        """
        加载缓存的数据集。

        :param training: 如果为 True 加载训练部分，否则加载测试部分。
        :param dataset_file: 数据集文件的路径。
        :return: 加载后的 M4Dataset 对象。
        """
        info_file = os.path.join(dataset_file, 'M4-info.csv')  # 数据集的基本信息文件路径
        train_cache_file = os.path.join(dataset_file, 'training.npz')  # 训练数据缓存文件路径
        test_cache_file = os.path.join(dataset_file, 'test.npz')  # 测试数据缓存文件路径
        m4_info = pd.read_csv(info_file)  # 读取 M4 数据集的信息文件
        return M4Dataset(ids=m4_info.M4id.values,
                         groups=m4_info.SP.values,
                         frequencies=m4_info.Frequency.values,
                         horizons=m4_info.Horizon.values,
                         values=np.load(
                             train_cache_file if training else test_cache_file,
                             allow_pickle=True))  # 加载训练或测试数据，并创建 M4Dataset 实例


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']  # 季节性模式
    horizons = [6, 8, 18, 13, 14, 48]  # 预测范围
    frequencies = [1, 4, 12, 1, 1, 24]  # 数据频率
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }  # 各种季节性模式对应的预测范围
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }  # 各种季节性模式对应的数据频率
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }  # 历史数据大小，根据 interpretable.gin 配置


def load_m4_info() -> pd.DataFrame:
    """
    加载 M4Info 文件。

    :return: 包含 M4Info 的 Pandas DataFrame。
    """
    return pd.read_csv(INFO_FILE_PATH)  # 读取并返回 M4Info 文件的内容作为 DataFrame
