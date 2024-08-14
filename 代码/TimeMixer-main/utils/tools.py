# 导入NumPy库，用于数值计算
import numpy as np

# 导入Pandas库，用于处理数据
import pandas as pd

# 导入PyTorch库，用于深度学习
import torch

# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt

# 将matplotlib的后端切换为非交互式后端'agg'，适合在没有图形界面的环境下生成图片
plt.switch_backend('agg')


# 定义调整学习率的函数
def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    """
    调整学习率的函数，根据不同的策略调整优化器中的学习率

    参数:
    ----------
    optimizer: torch.optim.Optimizer
        优化器对象，用于更新模型参数
    scheduler: torch.optim.lr_scheduler
        学习率调度器对象，可能用来动态调整学习率
    epoch: int
        当前训练的轮数
    args: dotdict
        包含各种训练超参数的字典对象
    printout: bool, 默认为True
        是否打印调整后的学习率信息
    """
    
    # 根据不同的学习率调整策略选择合适的学习率
    if args.lradj == 'type1':
        # 每个epoch减半
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # 指定的特定epoch对应特定学习率
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        # 前三次保持原始学习率，之后每次减0.9倍
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        # 每个epoch减5%
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        # 使用调度器返回的最后一次学习率
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    # 如果当前epoch在预定义的调整策略中
    if epoch in lr_adjust.keys():
        # 获取对应的学习率
        lr = lr_adjust[epoch]
        # 遍历优化器中的参数组，更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 如果需要打印输出，则打印当前的学习率
        if printout: print('Updating learning rate to {}'.format(lr))


# 定义EarlyStopping类，用于早停机制
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        初始化EarlyStopping对象

        参数:
        ----------
        patience: int, 默认为7
            如果验证损失在`patience`轮内没有改善，触发早停
        verbose: bool, 默认为False
            如果为True，则在每次保存模型时打印消息
        delta: float, 默认为0
            最小的改善量，如果小于该值则认为没有改善
        """
        self.patience = patience  # 记录容忍的轮数
        self.verbose = verbose  # 是否打印消息
        self.counter = 0  # 记录没有改善的次数
        self.best_score = None  # 保存最佳得分
        self.early_stop = False  # 记录是否应该早停
        self.val_loss_min = np.Inf  # 初始化最小验证损失为正无穷
        self.delta = delta  # 保存最小改善量

    def __call__(self, val_loss, model, path):
        """
        调用函数，实现早停逻辑

        参数:
        ----------
        val_loss: float
            当前epoch的验证损失
        model: torch.nn.Module
            需要保存的模型
        path: str
            保存模型的路径
        """
        score = -val_loss  # 得分设为验证损失的负值（损失越小，得分越高）
        
        if self.best_score is None:
            # 如果这是第一次调用，直接保存当前模型
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # 如果当前得分没有比之前最佳得分好，增加counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # 如果counter超过容忍度，触发早停
                self.early_stop = True
        else:
            # 如果当前得分好于之前最佳得分，保存模型并重置counter
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        保存模型到指定路径

        参数:
        ----------
        val_loss: float
            当前epoch的验证损失
        model: torch.nn.Module
            需要保存的模型
        path: str
            保存模型的路径
        """
        if self.verbose:
            # 如果verbose为True，打印保存消息
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 保存模型的状态字典到指定路径
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        # 更新最小验证损失
        self.val_loss_min = val_loss


# 定义dotdict类，允许通过点号来访问字典的属性
class dotdict(dict):
    """通过点号访问字典属性"""
    __getattr__ = dict.get  # 重载获取属性方法
    __setattr__ = dict.__setitem__  # 重载设置属性方法
    __delattr__ = dict.__delitem__  # 重载删除属性方法


# 定义标准化类，用于数据标准化处理
class StandardScaler():
    def __init__(self, mean, std):
        """
        初始化StandardScaler对象

        参数:
        ----------
        mean: float
            数据的均值
        std: float
            数据的标准差
        """
        self.mean = mean  # 记录数据均值
        self.std = std  # 记录数据标准差

    def transform(self, data):
        """
        标准化数据

        参数:
        ----------
        data: np.ndarray or torch.Tensor
            需要标准化的数据

        返回:
        ----------
        np.ndarray or torch.Tensor:
            标准化后的数据
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        逆标准化数据

        参数:
        ----------
        data: np.ndarray or torch.Tensor
            需要逆标准化的数据

        返回:
        ----------
        np.ndarray or torch.Tensor:
            逆标准化后的数据
        """
        return (data * self.std) + self.mean


# 定义函数用于保存预测结果为CSV文件
def save_to_csv(true, preds=None, name='./pic/test.pdf'):
    """
    将真实值和预测值保存为CSV文件

    参数:
    ----------
    true: np.ndarray or pd.Series
        真实值
    preds: np.ndarray or pd.Series, 默认为None
        预测值
    name: str, 默认为'./pic/test.pdf'
        保存文件的路径和名称
    """
    # 创建包含真实值和预测值的DataFrame
    data = pd.DataFrame({'true': true, 'preds': preds})
    # 将DataFrame保存为CSV文件
    data.to_csv(name, index=False, sep=',')


# 定义函数用于可视化真实值和预测值
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    将真实值和预测值进行可视化

    参数:
    ----------
    true: np.ndarray or pd.Series
        真实值
    preds: np.ndarray or pd.Series, 默认为None
        预测值
    name: str, 默认为'./pic/test.pdf'
        保存图片的路径和名称
    """
    plt.figure()  # 创建新图形
    plt.plot(true, label='GroundTruth', linewidth=2)  # 绘制真实值曲线
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)  # 如果有预测值，绘制预测值曲线
    plt.legend()  # 显示图例
    plt.savefig(name, bbox_inches='tight')  # 保存图片，去除多余空白


# 定义函数用于可视化权重矩阵
def visual_weights(weights, name='./pic/test.pdf'):
    """
    将权重矩阵进行可视化

    参数:
    ----------
    weights: np.ndarray
        权重矩阵
    name: str, 默认为'./pic/test.pdf'
        保存图片的路径和名称
    """
    fig, ax = plt.subplots()  # 创建子图
    # 使用imshow绘制权重矩阵，选择合适的颜色映射
    im = ax.imshow(weights, cmap='YlGnBu')
    # 添加颜色条到顶部
    fig.colorbar(im, pad=0.03, location='top')
    # 保存图片，高分辨率输出
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()  # 关闭图形


# 定义函数用于调整预测结果，使其更接近真实情况
def adjustment(gt, pred):
    """
    调整预测结果中的异常，使其更符合实际情况

    参数:
    ----------
    gt: np.ndarray
        真实标签
    pred: np.ndarray
        预测标签

    返回:
    ----------
    tuple: (gt, pred)
        调整后的真实标签和预测标签
    """
    anomaly_state = False  # 标记是否处于异常状态
    for i in range(len(gt)):
        # 检查是否进入异常状态
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True  # 更新为异常状态
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break  # 若遇到正常状态，停止向前修正
                else:
                    if pred[j] == 0:
                        pred[j] = 1  # 向前修正预测标签为异常
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break  # 若遇到正常状态，停止向后修正
                else:
                    if pred[j] == 0:
                        pred[j] = 1  # 向后修正预测标签为异常
        elif gt[i] == 0:
            anomaly_state = False  # 重置异常状态
        if anomaly_state:
            pred[i] = 1  # 如果在异常状态中，修正当前预测标签
    return gt, pred  # 返回修正后的标签


# 定义计算准确率的函数
def cal_accuracy(y_pred, y_true):
    """
    计算预测结果的准确率

    参数:
    ----------
    y_pred: np.ndarray
        预测标签
    y_true: np.ndarray
        真实标签

    返回:
    ----------
    float:
        准确率
    """
    return np.mean(y_pred == y_true)  # 返回预测结果与真实标签相等的比例
