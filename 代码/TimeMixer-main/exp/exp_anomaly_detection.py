from torch.optim import lr_scheduler  # 从 PyTorch 导入学习率调度器，用于动态调整优化器的学习率

from data_provider.data_factory import data_provider  # 从自定义模块中导入数据提供函数，用于获取训练和测试数据
from exp.exp_basic import Exp_Basic  # 从自定义模块中导入基本实验类，定义了实验的基本流程和方法
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment  # 从自定义工具模块中导入早停机制、学习率调整和其他辅助函数
from sklearn.metrics import precision_recall_fscore_support  # 从 scikit-learn 导入精度、召回率、F1 分数等指标的计算函数
from sklearn.metrics import accuracy_score  # 从 scikit-learn 导入准确率计算函数
import torch.multiprocessing  # 导入 PyTorch 的多进程模块，用于处理多进程任务

# 设置 PyTorch 的多进程共享策略为 'file_system'，以确保跨进程的张量数据共享
torch.multiprocessing.set_sharing_strategy('file_system')

import torch  # 导入 PyTorch 库，这是一个用于深度学习的开源框架
import torch.nn as nn  # 从 PyTorch 导入神经网络模块，提供神经网络层的构建和其他工具
from torch import optim  # 从 PyTorch 导入优化器模块，用于定义和应用优化算法
import os  # 导入操作系统模块，用于与操作系统进行交互，如文件路径处理
import time  # 导入时间模块，用于处理时间相关的操作，如计时
import warnings  # 导入警告模块，用于控制是否显示 Python 警告信息
import numpy as np  # 导入 NumPy 库，用于数值计算和数组操作

# 忽略所有警告信息，以减少控制台输出的噪声
warnings.filterwarnings('ignore')



# 定义异常检测实验类，继承自基础实验类 Exp_Basic
class Exp_Anomaly_Detection(Exp_Basic):
    # 初始化方法，接收实验参数 args
    def __init__(self, args): 
        # 调用父类 Exp_Basic 的初始化方法，传入参数 args
        super(Exp_Anomaly_Detection, self).__init__(args)

    # 构建模型的方法
    def _build_model(self):
        # 根据指定的模型名称，从模型字典中构建模型实例，并将模型的所有参数转换为浮点类型
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 判断是否配置了使用多 GPU 并且当前设备是否支持 GPU 计算
        if self.args.use_multi_gpu and self.args.use_gpu:
            # 如果满足条件，则使用 DataParallel 将模型并行化到指定的多个 GPU 设备上
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
    
        # 返回构建好的模型
        return model

    # 获取数据的方法
    def _get_data(self, flag):
        # 调用数据提供函数 data_provider，根据传入的参数 args 和标志 flag 加载数据集和数据加载器
        data_set, data_loader = data_provider(self.args, flag)
    
        # 返回加载好的数据集和数据加载器
        return data_set, data_loader

    # 选择优化器的方法
    def _select_optimizer(self):
        # 使用 Adam 优化器，并设置学习率为配置中的学习率，针对模型的参数进行优化
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        # 返回配置好的优化器
        return model_optim

    # 选择损失函数的方法
    def _select_criterion(self):
        # 选择均方误差损失函数 (MSELoss) 作为模型的损失函数
        criterion = nn.MSELoss()
        
        # 返回配置好的损失函数
        return criterion

    # 验证模型的方法
    def vali(self, vali_data, vali_loader, criterion):
        # 初始化总损失列表，用于存储每个批次的损失值
        total_loss = []
    
        # 将模型设置为评估模式，以禁用 dropout 等训练时特定的层
        self.model.eval()
    
        # 在验证过程中不需要计算梯度，因此使用 no_grad 来节省内存并加速计算
        with torch.no_grad():
            # 遍历验证数据加载器中的每个批次数据
            for i, (batch_x, _) in enumerate(vali_loader):
                # 将输入数据转换为浮点型并移动到指定的设备上（如 GPU）
                batch_x = batch_x.float().to(self.device)

                # 将输入数据传入模型进行预测，传入额外的 None 参数以匹配模型接口
                outputs = self.model(batch_x, None, None, None)

                # 根据特征类型选择输出维度：如果 features 是 'MS'，则使用倒数第一维；否则使用第一维
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
            
                # 将模型的输出与真实值分离以避免影响梯度计算
                pred = outputs.detach()
                true = batch_x.detach()

                # 使用传入的损失函数计算当前批次的损失
                loss = criterion(pred, true)
                
                # 将损失值添加到总损失列表中
                total_loss.append(loss.item())
    
        # 计算所有批次损失的平均值
        total_loss = np.average(total_loss)
    
        # 恢复模型为训练模式
        self.model.train()
    
        # 返回验证集上的平均损失值
        return total_loss



    # 定义训练方法，接收参数 setting（实验设置）
def train(self, setting):
        # 获取训练集、验证集和测试集及其对应的数据加载器
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 定义模型检查点的保存路径，如果路径不存在则创建该路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 记录当前时间，用于计算训练速度
        time_now = time.time()

        # 计算每个 epoch 的训练步数（即批次数量）
        train_steps = len(train_loader)

        # 初始化早停机制，设定耐心值和日志输出选项
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择优化器和损失函数
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 初始化学习率调度器，使用 OneCycleLR 策略调整学习率
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # 开始训练循环，遍历每个 epoch
        for epoch in range(self.args.train_epochs):
            iter_count = 0  # 迭代计数器
            train_loss = []  # 用于记录每个批次的训练损失

            # 将模型设置为训练模式
            self.model.train()

            # 记录每个 epoch 的起始时间
            epoch_time = time.time()

            # 遍历训练数据加载器中的每个批次
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()  # 清空优化器中的梯度

                # 将输入数据转换为浮点型并移动到指定的设备上（如 GPU）
                batch_x = batch_x.float().to(self.device)

                # 将输入数据传入模型进行预测
                outputs = self.model(batch_x, None, None, None)

                # 根据特征类型选择输出维度：如果 features 是 'MS'，则使用倒数第一维；否则使用第一维
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # 计算当前批次的损失，并将其记录到 train_loss 列表中
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                # 每训练 100 个批次输出一次日志信息
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 计算每个迭代的速度，并预测剩余时间
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0  # 重置迭代计数器
                    time_now = time.time()  # 更新当前时间

                # 反向传播计算梯度并更新模型参数
                loss.backward()
                model_optim.step()

                # 如果使用 TST 方式调整学习率，调用 adjust_learning_rate 函数调整学习率，并更新调度器
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # 输出每个 epoch 的耗时
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # 计算并记录该 epoch 的平均训练损失
            train_loss = np.average(train_loss)

            # 在验证集和测试集上进行验证，并计算损失
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 输出当前 epoch 的训练、验证和测试损失
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 使用早停机制监控测试损失，保存模型并在损失不再改善时提前停止训练
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 如果不使用 TST 方式调整学习率，调用 adjust_learning_rate 函数并输出新学习率
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # 加载训练期间保存的最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 返回训练好的模型
        return self.model

# 定义测试方法，接收参数 setting（实验设置）和 test（测试标志）
def test(self, setting, test=0):
        # 获取测试集和训练集及其加载器
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        # 如果 test 参数为 1，则加载预训练模型的权重
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []  # 用于存储训练集上的异常得分
        folder_path = './test_results/' + setting + '/'
        
        # 如果保存测试结果的文件夹不存在，则创建该文件夹
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 将模型设置为评估模式
        self.model.eval()
        # 定义用于异常检测的损失函数（不进行reduce操作）
        self.anomaly_criterion = nn.MSELoss(reduce=False)  

        # (1) 在训练集上统计异常得分
        with torch.no_grad():  # 禁用梯度计算
            # 遍历训练数据加载器中的每个批次
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)  # 将数据转换为浮点型并移动到指定设备上
                outputs = self.model(batch_x, None, None, None)  # 模型进行重构
                # 计算每个样本的平均异常得分
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()  # 将得分转换为 NumPy 数组
                attens_energy.append(score)  # 将得分添加到列表中

        # 将所有得分拼接为一个向量
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)  
        # 转换为 NumPy 数组保存训练集的异常得分
        train_energy = np.array(attens_energy)  

        # (2) 寻找阈值
        attens_energy = []  # 重置异常得分列表
        test_labels = []  # 用于存储测试集的标签
        # 遍历测试数据加载器中的每个批次
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)  # 将测试数据转换为浮点型并移动到指定设备上
            outputs = self.model(batch_x, None, None, None)  # 模型进行重构
            # 计算每个样本的平均异常得分
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()  # 将得分转换为 NumPy 数组
            attens_energy.append(score)  # 将得分添加到列表中
            test_labels.append(batch_y)  # 添加对应的测试标签

        # 将所有测试集的异常得分拼接为一个向量
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)  
        # 转换为 NumPy 数组保存测试集的异常得分
        test_energy = np.array(attens_energy)  
        # 将训练集和测试集的异常得分组合在一起
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)  
        # 根据异常比例计算阈值
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)  
        print("Threshold :", threshold)

        # (3) 在测试集上进行评估
        # 将得分高于阈值的样本预测为异常（1），否则为正常（0）
        pred = (test_energy > threshold).astype(int)  
        # 将测试标签拼接为一个向量
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)  
        # 转换为 NumPy 数组
        test_labels = np.array(test_labels)  
        gt = test_labels.astype(int)  # 将标签转换为整数类型

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) 检测结果的调整
        gt, pred = adjustment(gt, pred)  # 调整预测结果和实际标签

        pred = np.array(pred)  # 转换为 NumPy 数组
        gt = np.array(gt)  # 转换为 NumPy 数组
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        # 计算并输出评估指标：准确率、精确率、召回率和 F1 分数
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))

        # 将评估结果写入文件中保存
        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()

        return  # 返回函数，结束测试过程


