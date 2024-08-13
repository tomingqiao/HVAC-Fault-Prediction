from torch.optim import lr_scheduler  # 从 torch.optim 导入学习率调度器模块

from data_provider.data_factory import data_provider  # 从数据提供模块导入数据提供函数
from exp.exp_basic import Exp_Basic  # 导入基础实验类
from utils.tools import EarlyStopping, adjust_learning_rate, visual  # 导入工具函数：早停机制、学习率调整和可视化函数
from utils.metrics import metric  # 从 utils.metrics 导入评价指标函数
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torch import optim  # 从 PyTorch 导入优化器模块
import os  # 导入操作系统模块，用于文件和目录操作
import time  # 导入时间模块，用于记录和计算时间
import warnings  # 导入警告模块，用于处理警告信息
import numpy as np  # 导入 NumPy 库，用于科学计算和数组操作

warnings.filterwarnings('ignore')  # 忽略所有警告信息



class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)  # 调用父类的构造函数初始化实验类

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()  # 根据传入的模型名称从模型字典中获取模型并实例化

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)  # 如果使用多GPU并且启用了GPU，使用DataParallel进行模型并行化
        return model  # 返回构建好的模型

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)  # 根据标志(flag)获取相应的数据集和数据加载器
        return data_set, data_loader  # 返回数据集和数据加载器

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)  # 选择Adam优化器并设置学习率
        return model_optim  # 返回优化器

    def _select_criterion(self):
        criterion = nn.MSELoss()  # 选择均方误差(MSE)作为损失函数
        return criterion  # 返回损失函数

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []  # 存储总损失
        self.model.eval()  # 将模型设置为评估模式，禁用dropout等
        with torch.no_grad():  # 禁用梯度计算，减少内存消耗
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)  # 将输入数据转换为浮点类型并移动到指定设备
                batch_x_mark = batch_x_mark.float().to(self.device)  # 将时间戳数据转换为浮点类型并移动到指定设备

                # 随机掩码
                B, T, N = batch_x.shape  # 获取batch_x的形状(B为batch size, T为序列长度, N为特征数量)
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)  # 生成一个与batch_x相同形状的随机掩码张量
                mask[mask <= self.args.mask_rate] = 0  # 掩码率以下的元素设为0，表示掩盖
                mask[mask > self.args.mask_rate] = 1  # 掩码率以上的元素设为1，表示保留
                inp = batch_x.masked_fill(mask == 0, 0)  # 用0填充被掩盖的部分

                outputs = self.model(inp, batch_x_mark, None, None, mask)  # 将掩盖后的输入数据和时间戳输入模型

                f_dim = -1 if self.args.features == 'MS' else 0  # 根据特征类型选择输出的维度
                outputs = outputs[:, :, f_dim:]  # 取出输出中相应维度的部分

                # 支持多尺度(MS)
                batch_x = batch_x[:, :, f_dim:]  # 同样对输入数据取相应维度的部分
                mask = mask[:, :, f_dim:]  # 对掩码取相应维度的部分

                pred = outputs.detach()  # 分离输出，避免梯度计算
                true = batch_x.detach()  # 分离真实值，避免梯度计算
                mask = mask.detach()  # 分离掩码，避免梯度计算

                loss = criterion(pred[mask == 0], true[mask == 0])  # 计算被掩盖部分的损失
                total_loss.append(loss.item())  # 将损失值添加到总损失列表中
        total_loss = np.average(total_loss)  # 计算平均损失
        self.model.train()  # 恢复模型为训练模式
        return total_loss  # 返回验证集上的总损失


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')  # 获取训练数据和数据加载器
        vali_data, vali_loader = self._get_data(flag='val')  # 获取验证数据和数据加载器
        test_data, test_loader = self._get_data(flag='test')  # 获取测试数据和数据加载器

        path = os.path.join(self.args.checkpoints, setting)  # 设置模型检查点路径
        if not os.path.exists(path):
            os.makedirs(path)  # 如果路径不存在，创建目录

        time_now = time.time()  # 记录当前时间

        train_steps = len(train_loader)  # 获取训练步骤数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 初始化早停对象

        model_optim = self._select_optimizer()  # 选择优化器
        criterion = self._select_criterion()  # 选择损失函数

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)  # 设置学习率调度器

        for epoch in range(self.args.train_epochs):
            iter_count = 0  # 迭代计数
            train_loss = []  # 存储训练损失

            self.model.train()  # 将模型设置为训练模式
            epoch_time = time.time()  # 记录当前时间作为epoch开始时间
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()  # 清空梯度

                batch_x = batch_x.float().to(self.device)  # 将输入数据转换为浮点类型并移动到指定设备
                batch_x_mark = batch_x_mark.float().to(self.device)  # 同上，处理时间戳数据

                # 随机掩码
                B, T, N = batch_x.shape  # 获取输入数据的形状 (B=batch size, T=序列长度, N=特征数)
                mask = torch.rand((B, T, N)).to(self.device)  # 生成随机掩码
                mask[mask <= self.args.mask_rate] = 0  # 掩码率以下的部分被掩盖
                mask[mask > self.args.mask_rate] = 1  # 掩码率以上的部分被保留
                inp = batch_x.masked_fill(mask == 0, 0)  # 用0填充被掩盖的部分

                outputs = self.model(inp, batch_x_mark, None, None, mask)  # 将处理后的数据输入模型

                f_dim = -1 if self.args.features == 'MS' else 0  # 根据特征类型选择维度
                outputs = outputs[:, :, f_dim:]  # 取出相应维度的输出

                # 支持多尺度(MS)
                batch_x = batch_x[:, :, f_dim:]  # 同样对输入数据进行处理
                mask = mask[:, :, f_dim:]  # 处理掩码

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])  # 计算损失，仅考虑被掩盖部分
                train_loss.append(loss.item())  # 将损失值添加到训练损失列表中

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))  # 打印迭代信息
                    speed = (time.time() - time_now) / iter_count  # 计算每次迭代的时间
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)  # 计算剩余时间
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()  # 反向传播计算梯度
                model_optim.step()  # 更新模型参数

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)  # 调整学习率
                    scheduler.step()  # 更新学习率调度器

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))  # 打印当前epoch的耗时
            train_loss = np.average(train_loss)  # 计算平均训练损失
            vali_loss = self.vali(vali_data, vali_loader, criterion)  # 验证集上计算损失
            test_loss = self.vali(test_data, test_loader, criterion)  # 测试集上计算损失

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))  # 打印训练、验证和测试损失
            early_stopping(test_loss, self.model, path)  # 早停检查
            if early_stopping.early_stop:
                print("Early stopping")  # 如果早停条件满足，则停止训练
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)  # 调整学习率
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))  # 打印更新后的学习率

        best_model_path = path + '/' + 'checkpoint.pth'  # 设置最佳模型路径
        self.model.load_state_dict(torch.load(best_model_path))  # 加载最佳模型

        return self.model  # 返回训练好的模型

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')  # 获取测试数据和数据加载器
        if test:
            print('loading model')  # 如果需要加载模型，打印提示
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))  # 加载模型

        preds = []  # 存储预测结果
        trues = []  # 存储真实值
        masks = []  # 存储掩码
        folder_path = './test_results/' + setting + '/'  # 设置测试结果存储路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # 如果路径不存在，创建目录

        self.model.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 禁用梯度计算
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)  # 将输入数据转换为浮点类型并移动到指定设备
                batch_x_mark = batch_x_mark.float().to(self.device)  # 同上，处理时间戳数据

                # 随机掩码
                B, T, N = batch_x.shape  # 获取输入数据的形状
                mask = torch.rand((B, T, N)).to(self.device)  # 生成随机掩码
                mask[mask <= self.args.mask_rate] = 0  # 掩码率以下的部分被掩盖
                mask[mask > self.args.mask_rate] = 1  # 掩码率以上的部分被保留
                inp = batch_x.masked_fill(mask == 0, 0)  # 用0填充被掩盖的部分

                # 插补
                outputs = self.model(inp, batch_x_mark, None, None, mask)  # 将处理后的数据输入模型

                # 评估
                f_dim = -1 if self.args.features == 'MS' else 0  # 根据特征类型选择维度
                outputs = outputs[:, :, f_dim:]  # 取出相应维度的输出

                # 支持多尺度(MS)
                batch_x = batch_x[:, :, f_dim:]  # 同样对输入数据进行处理
                mask = mask[:, :, f_dim:]  # 处理掩码

                outputs = outputs.detach().cpu().numpy()  # 分离输出，转换为numpy数组
                pred = outputs  # 预测结果
                true = batch_x.detach().cpu().numpy()  # 真实值
                preds.append(pred)  # 存储预测结果
                trues.append(true)  # 存储真实值
                masks.append(mask.detach().cpu())  # 存储掩码

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()  # 复制真实值
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())  # 用预测结果填充掩盖部分
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))  # 可视化

        preds = np.concatenate(preds, 0)  # 将预测结果拼接在一起
        trues = np.concatenate(trues, 0)  # 将真实值拼接在一起
        masks = np.concatenate(masks, 0)  # 将掩码拼接在一起
        print('test shape:', preds.shape, trues.shape)  # 打印测试结果的形状

        # 保存结果
        folder_path = './results/' + setting + '/'  # 设置结果保存路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # 如果路径不存在，创建目录

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])  # 计算评估指标
        print('mse:{}, mae:{}'.format(mse, mae))  # 打印MSE和MAE
        f = open("result_imputation.txt", 'a')  # 打开结果文件（追加模式）
        f.write(setting + "  \n")  # 写入设置信息
        f.write('mse:{}, mae:{}'.format(mse, mae))  # 写入MSE和MAE
        f.write('\n')
        f.write('\n')
        f.close()  # 关闭文件

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))  # 保存评估指标
        np.save(folder_path + 'pred.npy', preds)  # 保存预测结果
        np.save(folder_path + 'true.npy', trues)  # 保存真实值
        return  # 返回

