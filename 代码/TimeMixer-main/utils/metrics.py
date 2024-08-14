# 导入NumPy库，用于进行数值计算和数组操作
import numpy as np

# 定义RSE函数，用于计算相对平方误差
def RSE(pred, true):
    """
    计算相对平方误差（Relative Squared Error）。
    
    参数:
    - pred: 预测值，类型为NumPy数组
    - true: 实际值，类型为NumPy数组
    
    返回:
    - 相对平方误差，类型为浮点数
    """
    # 计算预测值与实际值之间的平方误差总和的平方根
    numerator = np.sqrt(np.sum((true - pred) ** 2))
    # 计算实际值与其均值之间的平方误差总和的平方根
    denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
    # 返回相对平方误差
    return numerator / denominator

# 定义CORR函数，用于计算皮尔逊相关系数
def CORR(pred, true):
    """
    计算皮尔逊相关系数（Pearson Correlation Coefficient）。
    
    参数:
    - pred: 预测值，类型为NumPy数组
    - true: 实际值，类型为NumPy数组
    
    返回:
    - 皮尔逊相关系数，类型为浮点数
    """
    # 计算预测值与实际值的协方差
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    # 计算预测值和实际值的标准差乘积
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    # 返回皮尔逊相关系数的平均值
    return (u / d).mean(-1)

# 定义MAE函数，用于计算平均绝对误差
def MAE(pred, true):
    """
    计算平均绝对误差（Mean Absolute Error）。
    
    参数:
    - pred: 预测值，类型为NumPy数组
    - true: 实际值，类型为NumPy数组
    
    返回:
    - 平均绝对误差，类型为浮点数
    """
    # 计算预测值与实际值之间的绝对误差的平均值
    return np.mean(np.abs(pred - true))

# 定义MSE函数，用于计算均方误差
def MSE(pred, true):
    """
    计算均方误差（Mean Squared Error）。
    
    参数:
    - pred: 预测值，类型为NumPy数组
    - true: 实际值，类型为NumPy数组
    
    返回:
    - 均方误差，类型为浮点数
    """
    # 计算预测值与实际值之间的平方误差的平均值
    return np.mean((pred - true) ** 2)

# 定义RMSE函数，用于计算均方根误差
def RMSE(pred, true):
    """
    计算均方根误差（Root Mean Squared Error）。
    
    参数:
    - pred: 预测值，类型为NumPy数组
    - true: 实际值，类型为NumPy数组
    
    返回:
    - 均方根误差，类型为浮点数
    """
    # 计算均方误差的平方根
    return np.sqrt(MSE(pred, true))

# 定义MAPE函数，用于计算平均绝对百分比误差
def MAPE(pred, true):
    """
    计算平均绝对百分比误差（Mean Absolute Percentage Error）。
    
    参数:
    - pred: 预测值，类型为NumPy数组
    - true: 实际值，类型为NumPy数组
    
    返回:
    - 平均绝对百分比误差，类型为浮点数
    """
    # 计算预测值与实际值之间的相对误差的绝对值
    mape = np.abs((pred - true) / true)
    # 将超过阈值的误差设为0，以避免异常值的影响
    mape = np.where(mape > 5, 0, mape)
    # 返回平均绝对百分比误差
    return np.mean(mape)

# 定义MSPE函数，用于计算均方百分比误差
def MSPE(pred, true):
    """
    计算均方百分比误差（Mean Squared Percentage Error）。
    
    参数:
    - pred: 预测值，类型为NumPy数组
    - true: 实际值，类型为NumPy数组
    
    返回:
    - 均方百分比误差，类型为浮点数
    """
    # 计算预测值与实际值之间的相对误差的平方
    return np.mean(np.square((pred - true) / true))

# 定义metric函数，用于综合计算多种评估指标
def metric(pred, true):
    """
    计算多个评估指标，包括MAE、MSE、RMSE、MAPE和MSPE。
    
    参数:
    - pred: 预测值，类型为NumPy数组
    - true: 实际值，类型为NumPy数组
    
    返回:
    - 包含MAE、MSE、RMSE、MAPE和MSPE的元组
    """
    # 计算平均绝对误差
    mae = MAE(pred, true)
    # 计算均方误差
    mse = MSE(pred, true)
    # 计算均方根误差
    rmse = RMSE(pred, true)
    # 计算平均绝对百分比误差
    mape = MAPE(pred, true)
    # 计算均方百分比误差
    mspe = MSPE(pred, true)
    
    # 返回所有计算的指标
    return mae, mse, rmse, mape, mspe
