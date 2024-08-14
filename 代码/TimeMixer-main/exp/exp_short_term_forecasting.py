# 导入学习率调度器模块
from torch.optim import lr_scheduler

# 导入数据提供器和M4数据集元数据模块
from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta

# 导入实验基础类
from exp.exp_basic import Exp_Basic

# 导入工具模块，包括早停机制、学习率调整、可视化和保存到CSV的功能
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv

# 导入自定义损失函数，包括MAPE损失、MASE损失和SMAPE损失
from utils.losses import mape_loss, mase_loss, smape_loss

# 导入M4数据集总结模块
from utils.m4_summary import M4Summary

# 导入PyTorch及其神经网络模块
import torch
import torch.nn as nn

# 导入优化器模块
from torch import optim

# 导入操作系统模块，用于文件路径管理
import os

# 导入时间模块，用于计时
import time

# 导入警告模块，用于忽略警告信息
import warnings

# 导入NumPy用于数值计算
import numpy as np

# 导入Pandas用于数据处理
import pandas

# 忽略所有警告信息
warnings.filterwarnings('ignore')



# 定义一个短期预测实验类，继承自基础实验类
class Exp_Short_Term_Forecast(Exp_Basic):
    
    # 初始化函数，接受实验参数并调用父类的初始化函数
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)  # 调用父类的构造函数

    # 构建模型函数，设置模型相关参数并返回模型
    def _build_model(self):
        # 如果数据集是M4，则根据季节性模式设置预测长度和序列长度等参数
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # 设置预测长度
            self.args.seq_len = 2 * self.args.pred_len  # 输入序列长度为预测长度的两倍
            self.args.label_len = self.args.pred_len  # 标签长度设置为预测长度
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]  # 设置频率映射
        
        # 从模型字典中根据参数选择模型，并将模型参数转换为浮点类型
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 如果使用多GPU且启用了GPU，则使用DataParallel进行多GPU并行计算
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model  # 返回构建的模型

    # 获取数据函数，根据标志（训练、验证、测试）获取相应的数据集和数据加载器
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)  # 调用数据提供函数
        return data_set, data_loader  # 返回数据集和数据加载器

    # 选择优化器函数，使用Adam优化器并返回
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)  # Adam优化器
        return model_optim  # 返回优化器

    # 选择损失函数函数，根据损失名称选择并返回相应的损失函数
    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()  # 均方误差损失
        elif loss_name == 'MAPE':
            return mape_loss()  # 平均绝对百分比误差
        elif loss_name == 'MASE':
            return mase_loss()  # 平均绝对缩放误差
        elif loss_name == 'SMAPE':
            return smape_loss()  # 对称平均绝对百分比误差

    # 训练函数，用于模型的训练过程
    def train(self, setting):
        # 获取训练数据和验证数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # 设置模型检查点路径，如果路径不存在则创建路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()  # 记录当前时间

        train_steps = len(train_loader)  # 计算训练步骤数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 提前停止机制

        model_optim = self._select_optimizer()  # 选择优化器
        criterion = self._select_criterion(self.args.loss)  # 选择损失函数

        # 设置学习率调度器
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # 训练循环
        for epoch in range(self.args.train_epochs):
            iter_count = 0  # 迭代计数器
            train_loss = []  # 存储训练损失

            self.model.train()  # 设置模型为训练模式
            epoch_time = time.time()  # 记录epoch开始时间
            # 遍历训练数据
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1  # 更新迭代计数器
                model_optim.zero_grad()  # 梯度清零

                # 将输入数据和标签移动到设备（如GPU），并转换为浮点类型
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 创建解码器输入，将预测长度的零张量与标签拼接
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 前向传播，获取模型输出
                outputs = self.model(batch_x, None, dec_inp, None)
                f_dim = -1 if self.args.features == 'MS' else 0  # 特征维度选择
                outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 截取输出的最后一部分
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 标签数据处理
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 计算损失值
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                loss = loss_value  # 总损失
                train_loss.append(loss.item())  # 记录损失

                # 每100次迭代输出一次训练状态
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count  # 计算迭代速度
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)  # 估算剩余时间
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0  # 重置迭代计数器
                    time_now = time.time()  # 更新当前时间

                # 反向传播并更新模型参数
                loss.backward()
                model_optim.step()

                # 如果学习率调整策略为TST，则更新学习率
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # 输出当前epoch的耗时
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)  # 计算平均训练损失
            vali_loss = self.vali(train_loader, vali_loader, criterion)  # 计算验证损失
            test_loss = vali_loss  # 测试损失与验证损失相同
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # 提前停止机制
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")  # 如果验证损失不再降低，则提前停止训练
                break

            # 如果学习率调整策略不是TST，则手动调整学习率
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # 加载最佳模型的权重
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model  # 返回训练后的模型

    # 验证函数，在验证集上评估模型
    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()  # 获取最后一个训练窗口
        y = vali_loader.dataset.timeseries  # 获取验证集的时间序列
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)  # 扩展输入数据的维度

        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            # 解码器输入
            B, _, C = x.shape  # 获取输入形状
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()

            # 初始化输出张量，并分批处理验证集数据
            outputs = torch.zeros((B, self.args.pred_len, C)).float()
            id_list = np.arange(0, B, 500)  # 每500个数据点为一批
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                x_enc = x[id_list[i]:id_list[i + 1]]
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x_enc, None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]],
                                                                      None).detach().cpu()
            f_dim = -1 if self.args.features == 'MS' else 0  # 特征维度选择
            outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 截取输出的最后一部分
            pred = outputs  # 预测结果
            true = torch.from_numpy(np.array(y))  # 转换真实值
            batch_y_mark = torch.ones(true.shape)  # 创建标签标记

            # 计算验证损失
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()  # 恢复模型为训练模式
        return loss  # 返回验证损失


    def test(self, setting, test=0):  # 定义测试函数，接收两个参数：设置和测试标志
        # 获取训练数据和测试数据的加载器
        _, train_loader = self._get_data(flag='train')  # 获取训练数据加载器
        _, test_loader = self._get_data(flag='test')  # 获取测试数据加载器
        
        # 获取最后一个训练窗口的数据（x）和测试集的时间序列（y）
        x, _ = train_loader.dataset.last_insample_window()  # 获取最后一个训练窗口
        y = test_loader.dataset.timeseries  # 获取测试集的时间序列
        
        # 将x转换为浮点型张量，并移动到指定设备（如GPU）
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)  # 在最后一个维度上扩展x的维度

        # 如果test标志为1，则加载模型的权重
        if test:
            print('loading model')  # 打印加载模型信息
            # 加载模型权重
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 设置测试结果的保存路径，如果路径不存在则创建
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算，减少内存消耗，加快推理速度
            B, _, C = x.shape  # 获取输入张量x的形状（B：批量大小，C：通道数）
            # 创建解码器输入张量，初始化为零，并移动到指定设备
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            # 将x的最后一部分与解码器输入张量拼接，形成新的解码器输入
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            
            # 初始化输出张量，存储模型预测结果
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)  # 生成一个从0到B的等差数列，步长为1
            id_list = np.append(id_list, B)  # 将批量大小B添加到id列表中
            
            # 遍历id列表，并对每一部分数据进行模型预测
            for i in range(len(id_list) - 1):
                x_enc = x[id_list[i]:id_list[i + 1]]  # 获取当前批次的输入数据
                # 模型预测，将结果存储在outputs中
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x_enc, None, dec_inp[id_list[i]:id_list[i + 1]], None)

                # 每隔1000次输出当前id
                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            # 根据特征类型调整输出张量的维度
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 截取预测长度部分的输出
            outputs = outputs.detach().cpu().numpy()  # 将输出张量转换为NumPy数组

            # 将预测结果和真实值赋值给变量
            preds = outputs  # 预测结果
            trues = y  # 真实值
            x = x.detach().cpu().numpy()  # 将输入张量x转换为NumPy数组

            # 遍历预测结果，生成图表和CSV文件，保存到指定路径
            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)  # 生成真实值序列
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)  # 生成预测值序列
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))  # 可视化预测结果
                save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))  # 保存预测结果到CSV文件

        print('test shape:', preds.shape)  # 输出预测结果的形状

        # 设置M4结果的保存路径，如果路径不存在则创建
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 将预测结果保存为DataFrame，并保存为CSV文件
        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]  # 设置DataFrame的索引
        forecasts_df.index.name = 'id'  # 设置索引的名称
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)  # 将第一列设为索引
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')  # 保存DataFrame到CSV文件

        # 输出模型名称
        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'  # 设置M4结果的文件路径

        # 如果所有任务完成，计算平均指标并输出结果
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)  # 实例化M4总结类
            smape_results, owa_results, mape, mase = m4_summary.evaluate()  # 评估模型性能
            # 输出SMAPE、MAPE、MASE、OWA指标
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')  # 提示完成所有任务后可计算平均指标
        
        return  # 返回结果


