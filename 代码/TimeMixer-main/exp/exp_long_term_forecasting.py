from torch.optim import lr_scheduler  # 导入PyTorch的学习率调度器模块

from data_provider.data_factory import data_provider  # 从数据提供者模块导入data_provider函数
from exp.exp_basic import Exp_Basic  # 从实验基础模块导入Exp_Basic类
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights  # 从工具模块导入相关函数和类
from utils.metrics import metric  # 从指标模块导入metric函数
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch import optim  # 从PyTorch导入优化器模块
import os  # 导入操作系统模块，用于路径操作
import time  # 导入时间模块，用于记录时间和计算耗时
import warnings  # 导入警告模块，用于处理Python警告
import numpy as np  # 导入NumPy库，用于科学计算

warnings.filterwarnings('ignore')  # 忽略警告信息，避免控制台输出过多无关信息



class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)  # 调用父类的初始化方法

    def _build_model(self):
        # 根据传入的模型名称，从字典中获取模型类，并实例化为浮点数类型
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 如果使用多GPU并且使用GPU，使用DataParallel进行多GPU并行处理
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 根据标志获取数据集和数据加载器
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 选择优化器为Adam，并设置学习率
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 根据数据类型选择损失函数，如果是PEMS数据集使用L1Loss，否则使用MSELoss
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []  # 存储所有的损失
        self.model.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 关闭梯度计算
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 将数据转换为浮点数并移动到GPU设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 对于PEMS或Solar数据集，不使用标记数据
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                # 根据是否存在下采样层来构建解码器输入
                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder-decoder过程
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():  # 使用自动混合精度
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0  # 根据特征类型设置维度

                pred = outputs.detach()  # 分离预测结果
                true = batch_y.detach()  # 分离真实值

                # 对于PEMS数据集，反向转换预测值和真实值，并计算评价指标
                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    total_loss.append(mae)

                else:
                    loss = criterion(pred, true)  # 计算损失
                    total_loss.append(loss.item())  # 将损失值存储到列表中

        total_loss = np.average(total_loss)  # 计算平均损失
        self.model.train()  # 将模型重新设置为训练模式
        return total_loss  # 返回验证损失

    def train(self, setting):
        # 获取训练、验证和测试数据及其对应的加载器
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 创建检查点保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()  # 记录当前时间

        train_steps = len(train_loader)  # 计算训练步骤数量
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 实例化早停机制

        model_optim = self._select_optimizer()  # 选择优化器
        criterion = self._select_criterion()  # 选择损失函数

        # 配置学习率调度器
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 使用自动混合精度

        for epoch in range(self.args.train_epochs):  # 开始训练循环
            iter_count = 0
            train_loss = []

            self.model.train()  # 设置模型为训练模式
            epoch_time = time.time()  # 记录每个epoch开始时间

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()  # 优化器梯度清零

                # 将数据转换为浮点数并移动到GPU设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 对于PEMS或Solar数据集，不使用标记数据
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                # 根据是否存在下采样层来构建解码器输入
                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder-decoder过程
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():  # 使用自动混合精度
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 获取最后的预测结果
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)  # 计算损失
                        train_loss.append(loss.item())  # 存储训练损失
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    loss = criterion(outputs, batch_y)  # 计算损失
                    train_loss.append(loss.item())  # 存储训练损失

                if (i + 1) % 100 == 0:  # 每100次迭代输出一次信息
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count  # 计算每次迭代的速度
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)  # 估算剩余时间
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()  # 更新当前时间

                if self.args.use_amp:
                    scaler.scale(loss).backward()  # 使用自动混合精度反向传播
                    scaler.step(model_optim)  # 更新优化器参数
                    scaler.update()  # 更新scaler
                else:
                    loss.backward()  # 反向传播
                    model_optim.step()  # 更新优化器参数

                scheduler.step()  # 更新学习率

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)  # 计算平均训练损失
            vali_loss = self.vali(vali_data, vali_loader, criterion)  # 计算验证损失
            test_loss = self.vali(test_data, test_loader, criterion)  # 计算测试损失

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)  # 早停机制

            if early_stopping.early_stop:  # 如果满足早停条件，终止训练
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'  # 加载最好的模型
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=0):
        # 获取测试数据集和数据加载器
        test_data, test_loader = self._get_data(flag='test')

        # 如果传入的test标志为真，加载之前保存的模型
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 设置检查点路径
        checkpoints_path = './checkpoints/' + setting + '/'
        preds = []  # 用于存储预测结果
        trues = []  # 用于存储真实值

        # 设置测试结果保存路径
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # 如果路径不存在，则创建文件夹

        # 将模型设置为评估模式
        self.model.eval()
        with torch.no_grad():  # 关闭梯度计算
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # 将输入数据转换为浮点数并移动到GPU设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # 将时间标记数据转换为浮点数并移动到GPU设备
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 对于PEMS或Solar数据集，不使用时间标记数据
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                # 如果没有下采样层，构建解码器输入
                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder-decoder 过程
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():  # 使用自动混合精度
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0  # 根据特征类型选择维度

                outputs = outputs.detach().cpu().numpy()  # 将预测结果从GPU移动到CPU并转为numpy数组
                batch_y = batch_y.detach().cpu().numpy()  # 将真实值从GPU移动到CPU并转为numpy数组

                preds.append(outputs)  # 将预测结果加入列表
                trues.append(batch_y)  # 将真实值加入列表

                # 每隔20次迭代进行可视化保存
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()  # 将输入数据从GPU移动到CPU并转为numpy数组
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)  # 逆转换输入数据

                    # 拼接真实值和预测值进行对比
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)

                    # 保存对比图像
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)  # 将预测结果列表转为numpy数组
        trues = np.array(trues)  # 将真实值列表转为numpy数组
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])  # 调整预测结果形状
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])  # 调整真实值形状
        print('test shape:', preds.shape, trues.shape)

        # 对于PEMS数据集，进行逆转换
        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # 结果保存
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # 如果路径不存在，则创建文件夹

        # 计算各种评价指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))

        # 保存结果到文本文件中
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        if self.args.data == 'PEMS':
            f.write('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))
        else:
            f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # 保存评价指标和预测结果、真实值为numpy文件
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return  # 函数返回

