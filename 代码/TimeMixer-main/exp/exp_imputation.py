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



# 定义一个名为 Exp_Imputation 的类，继承自 Exp_Basic
class Exp_Imputation(Exp_Basic):
    # 初始化函数，接受参数 args
    def __init__(self, args):
        # 调用父类的初始化函数
        super(Exp_Imputation, self).__init__(args)

    # 构建模型的方法
    def _build_model(self):
        # 从模型字典中获取模型类并实例化，同时将模型转换为浮点类型
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 如果使用多GPU并且启用了GPU，使用DataParallel进行模型并行化
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # 返回构建好的模型
        return model

    # 获取数据的方法
    def _get_data(self, flag):
        # 根据标志(flag)获取相应的数据集和数据加载器
        data_set, data_loader = data_provider(self.args, flag)
        # 返回数据集和数据加载器
        return data_set, data_loader

    # 选择优化器的方法
    def _select_optimizer(self):
        # 使用Adam优化器，并设置学习率
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # 返回优化器
        return model_optim

    # 选择损失函数的方法
    def _select_criterion(self):
        # 使用均方误差(MSE)作为损失函数
        criterion = nn.MSELoss()
        # 返回损失函数
        return criterion

    # 验证模型的方法
    def vali(self, vali_data, vali_loader, criterion):
        # 用于存储总损失的列表
        total_loss = []
        # 设置模型为评估模式，禁用dropout等操作
        self.model.eval()
        # 禁用梯度计算，以减少内存消耗
        with torch.no_grad():
            # 遍历验证数据加载器中的每个batch
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 将输入数据转换为浮点类型并移动到指定设备
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # 随机掩码生成
                B, T, N = batch_x.shape  # 获取batch_x的形状
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)  # 在指定设备上生成一个与batch_x相同形状的随机掩码张量
                mask[mask <= self.args.mask_rate] = 0  # 掩码率以下的元素设为0，表示掩盖这些元素
                mask[mask > self.args.mask_rate] = 1  # 掩码率以上的元素设为1，表示保留这些元素
                inp = batch_x.masked_fill(mask == 0, 0)  # 用0填充被掩盖的部分

                # 通过模型进行前向传播
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # 根据特征类型调整输出和输入的维度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # 对输入数据和掩码进行相同的维度选择
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                # 计算损失
                pred = outputs.detach()  # 分离输出，避免梯度计算
                true = batch_x.detach()  # 分离真实值，避免梯度计算
                mask = mask.detach()  # 分离掩码，避免梯度计算

                loss = criterion(pred[mask == 0], true[mask == 0])  # 计算被掩盖部分的损失
                total_loss.append(loss.item())  # 将损失值添加到总损失列表中
        total_loss = np.average(total_loss)  # 计算平均损失
        self.model.train()  # 恢复模型为训练模式
        return total_loss  # 返回验证集上的总损失

    # 训练模型的方法
    def train(self, setting):
        # 获取训练集、验证集和测试集的数据和数据加载器
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 定义模型检查点的路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)  # 如果路径不存在，则创建路径

        time_now = time.time()  # 获取当前时间

        train_steps = len(train_loader)  # 获取训练步数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 初始化早停机制

        model_optim = self._select_optimizer()  # 选择优化器
        criterion = self._select_criterion()  # 选择损失函数

        # 设置学习率调度器
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # 开始训练循环
        for epoch in range(self.args.train_epochs):
            iter_count = 0  # 迭代计数
            train_loss = []  # 存储每个batch的损失

            self.model.train()  # 将模型设置为训练模式
            epoch_time = time.time()  # 记录epoch开始时间
            # 遍历训练数据加载器中的每个batch
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1  # 迭代计数加1
                model_optim.zero_grad()  # 清除梯度

                # 将输入数据转换为浮点类型并移动到指定设备
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # 随机掩码生成
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # 掩盖
                mask[mask > self.args.mask_rate] = 1  # 保留
                inp = batch_x.masked_fill(mask == 0, 0)

                # 通过模型进行前向传播
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # 根据特征类型调整输出和输入的维度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # 对输入数据和掩码进行相同的维度选择
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                # 计算损失
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())  # 将损失添加到列表中

                # 每100个迭代输出一次训练信息
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count  # 计算速度
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)  # 计算剩余时间
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0  # 重置迭代计数
                    time_now = time.time()  # 更新当前时间

                # 反向传播并更新参数
                loss.backward()
                model_optim.step()

                # 调整学习率
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # 输出每个epoch的耗时
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)  # 计算平均训练损失
            vali_loss = self.vali(vali_data, vali_loader, criterion)  # 计算验证集损失
            test_loss = self.vali(test_data, test_loader, criterion)  # 计算测试集损失

            # 输出每个epoch的损失信息
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 使用早停机制判断是否需要停止训练
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 保存训练完毕的模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 返回训练损失、验证损失和测试损失
        return self.model

    # 测试模型的方法
    def test(self, setting):
        # 获取测试集的数据和数据加载器
        test_data, test_loader = self._get_data(flag='test')

        # 定义模型检查点的路径
        path = os.path.join(self.args.checkpoints, setting)

        # 如果模型未训练，则加载保存的模型参数
        if not os.path.exists(path):
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        # 设置模型为评估模式
        self.model.eval()

        # 初始化评估指标
        preds = []  # 存储预测值
        trues = []  # 存储真实值
        masks = []  # 存储掩码

        # 禁用梯度计算，以减少内存消耗
        with torch.no_grad():
            # 遍历测试数据加载器中的每个batch
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # 将输入数据转换为浮点类型并移动到指定设备
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # 随机掩码生成
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # 掩盖
                mask[mask > self.args.mask_rate] = 1  # 保留
                inp = batch_x.masked_fill(mask == 0, 0)

                # 通过模型进行前向传播
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # 根据特征类型调整输出和输入的维度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]

                # 收集预测值、真实值和掩码
                preds.append(outputs.cpu().numpy())
                trues.append(batch_x.cpu().numpy())
                masks.append(mask.cpu().numpy())

        # 将预测值、真实值和掩码拼接为单个数组
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(masks, axis=0)

        # 计算评估指标
        mae, mse, rmse, mape, mspe = metric(preds, trues, masks)
        # 输出评估指标
        print('mse:{}, mae:{}'.format(mse, mae))

        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 将预测值、真实值和评估指标保存到文件中
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
