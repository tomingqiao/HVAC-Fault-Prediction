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



class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        # 初始化短期预测实验的类，并继承基础实验类
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # 构建模型
        if self.args.data == 'm4':
            # 根据M4配置设置预测长度、序列长度和标签长度
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
            self.args.seq_len = 2 * self.args.pred_len  # 输入长度为2倍的预测长度
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        
        # 根据指定的模型名称创建模型实例
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 如果使用多GPU和GPU，则将模型并行化
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model

    def _get_data(self, flag):
        # 获取数据集和数据加载器
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 选择优化器，这里使用Adam优化器
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        # 根据指定的损失函数名称选择损失函数
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        # 开始训练过程
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # 设置检查点保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()  # 记录当前时间

        train_steps = len(train_loader)  # 获取训练步骤数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 初始化早停机制

        model_optim = self._select_optimizer()  # 选择优化器
        criterion = self._select_criterion(self.args.loss)  # 选择损失函数

        # 设置学习率调度器
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []  # 存储每个epoch的训练损失

            self.model.train()  # 设置模型为训练模式
            epoch_time = time.time()  # 记录每个epoch的开始时间
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()  # 梯度清零

                # 将输入数据和目标数据转换为浮点数并移动到设备上（如GPU）
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构建解码器输入，填充预测长度部分
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 前向传播获取模型输出
                outputs = self.model(batch_x, None, dec_inp, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 计算损失
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                loss = loss_value  # 可以根据需要增加其他损失项
                train_loss.append(loss.item())  # 记录损失

                # 每100次迭代打印一次当前损失和训练速度
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()  # 反向传播
                model_optim.step()  # 优化器更新参数

                # 根据学习率调整策略调整学习率
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # 打印每个epoch的花费时间
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)  # 计算训练损失的平均值
            vali_loss = self.vali(train_loader, vali_loader, criterion)  # 验证集损失
            test_loss = vali_loss  # 测试集损失等同于验证集损失
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 早停机制
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率并打印更新信息
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model  # 返回训练后的模型


    def vali(self, train_loader, vali_loader, criterion):
        # 获取训练集中的最后一个样本窗口x和验证集中的时间序列y
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)  # 将x转换为张量并转移到设备上
        x = x.unsqueeze(-1)  # 增加一个维度

        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            # 准备解码器输入
            B, _, C = x.shape  # 获取批量大小和通道数
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)  # 初始化解码器输入
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()  # 拼接标签部分和解码器输入

            # 编码器 - 解码器
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # 初始化输出张量
            id_list = np.arange(0, B, 500)  # 划分验证集
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                x_enc = x[id_list[i]:id_list[i + 1]]  # 获取当前批次的输入
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x_enc, None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]],
                                                                      None).detach().cpu()  # 前向传播获取输出

            f_dim = -1 if self.args.features == 'MS' else 0  # 获取输出维度
            outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 只保留预测长度部分
            pred = outputs  # 预测结果
            true = torch.from_numpy(np.array(y))  # 将y转换为张量
            batch_y_mark = torch.ones(true.shape)  # 创建标记张量

            # 计算损失
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()  # 恢复模型为训练模式
        return loss  # 返回验证损失

def test(self, setting, test=0):
        # 获取训练数据和测试数据
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:  # 如果需要测试，则加载模型权重
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'  # 设置测试结果保存路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            B, _, C = x.shape  # 获取批次大小和通道数
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)  # 初始化解码器输入
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()  # 拼接标签部分和解码器输入

            # 编码器 - 解码器
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                x_enc = x[id_list[i]:id_list[i + 1]]  # 获取当前批次的输入
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x_enc, None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]], None)

                if id_list[i] % 1000 == 0:  # 每处理1000个样本打印一次进度
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0  # 获取输出维度
            outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 只保留预测长度部分
            outputs = outputs.detach().cpu().numpy()  # 转换为numpy数组

            preds = outputs  # 预测结果
            trues = y  # 真实值
            x = x.detach().cpu().numpy()  # 将x转换为numpy数组

            # 可视化预测结果
            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))

        print('test shape:', preds.shape)  # 打印测试结果形状

        # 保存结果
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)  # 打印模型名称
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            # 如果所有任务都已完成，则计算评估指标
            m4_summary = M4Summary(file_path, self.args.root_path)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return


