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



# 定义一个名为 Exp_Long_Term_Forecast 的类，继承自 Exp_Basic
class Exp_Long_Term_Forecast(Exp_Basic):
    # 初始化函数，接受参数 args
    def __init__(self, args):
        # 调用父类的初始化函数
        super(Exp_Long_Term_Forecast, self).__init__(args)

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
        # 根据数据集类型选择损失函数
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()  # 使用L1损失函数
        else:
            criterion = nn.MSELoss()  # 使用均方误差损失函数
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
                batch_y = batch_y.float().to(self.device)

                # 将时间标记数据转换为浮点类型并移动到指定设备
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 如果数据集是PEMS或Solar，不使用时间标记数据
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                # 根据下采样层数决定是否生成decoder输入
                if self.args.down_sampling_layers == 0:
                    # 生成一个与batch_y后n个时间步长相同形状的零张量，并与batch_y的前label_len个时间步长拼接
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # 编码器 - 解码器
                if self.args.use_amp:
                    # 如果使用自动混合精度进行前向传播
                    with torch.cuda.amp.autocast():
                        # 如果输出注意力图，则只取模型输出的第一个元素
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    # 正常进行前向传播
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # 根据特征类型调整输出和输入的维度
                f_dim = -1 if self.args.features == 'MS' else 0

                # 获取模型的预测值和真实值
                pred = outputs.detach()
                true = batch_y.detach()

                # 如果数据集是PEMS
                if self.args.data == 'PEMS':
                    B, T, C = pred.shape  # 获取预测值的形状
                    pred = pred.cpu().numpy()  # 将预测值移动到CPU并转换为numpy数组
                    true = true.cpu().numpy()  # 将真实值移动到CPU并转换为numpy数组
                    # 对预测值和真实值进行反归一化处理
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    # 计算评估指标
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    total_loss.append(mae)  # 将MAE损失添加到总损失列表中

                else:
                    # 计算损失
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())  # 将损失值添加到总损失列表中

        # 计算平均损失
        total_loss = np.average(total_loss)
        # 恢复模型为训练模式
        self.model.train()
        # 返回验证集上的总损失
        return total_loss


    # 训练函数，用于训练模型
    def train(self, setting):
        # 获取训练数据和对应的加载器
        train_data, train_loader = self._get_data(flag='train')
        # 获取验证数据和对应的加载器
        vali_data, vali_loader = self._get_data(flag='val')
        # 获取测试数据和对应的加载器
        test_data, test_loader = self._get_data(flag='test')

        # 设置模型保存路径
        path = os.path.join(self.args.checkpoints, setting)
        # 如果路径不存在，则创建路径
        if not os.path.exists(path):
            os.makedirs(path)

        # 记录当前时间
        time_now = time.time()

        # 计算训练步骤数
        train_steps = len(train_loader)
        # 初始化早停机制，patience表示在验证集上多少个epoch内损失不下降则停止训练
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择优化器
        model_optim = self._select_optimizer()
        # 选择损失函数
        criterion = self._select_criterion()

        # 初始化学习率调度器，使用OneCycleLR策略
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # 如果使用自动混合精度，则初始化梯度缩放器
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 训练的主循环，遍历每个epoch
        for epoch in range(self.args.train_epochs):
            # 初始化迭代计数
            iter_count = 0
            # 初始化训练损失列表
            train_loss = []

            # 将模型设置为训练模式
            self.model.train()
            # 记录当前epoch的开始时间
            epoch_time = time.time()

            # 遍历训练数据集的每个批次
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # 迭代计数加1
                iter_count += 1
                # 优化器梯度清零
                model_optim.zero_grad()

                # 将输入数据和标签转换为浮点类型并移动到设备上（如GPU）
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # 将时间标记数据转换为浮点类型并移动到设备上
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 如果数据集是'PEMS'或'Solar'，则不使用时间标记数据
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                # 如果不使用下采样层，则生成decoder的输入
                if self.args.down_sampling_layers == 0:
                    # 生成一个与batch_y后n个时间步长相同形状的零张量，并与batch_y的前label_len个时间步长拼接
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # 编码器 - 解码器部分
                if self.args.use_amp:
                    # 如果使用自动混合精度进行前向传播
                    with torch.cuda.amp.autocast():
                        # 如果输出注意力图，则只取模型输出的第一个元素
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # 根据特征类型调整输出的维度
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # 计算损失
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())  # 将损失值添加到训练损失列表中
                else:
                    # 正常进行前向传播
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # 根据特征类型调整输出的维度
                    f_dim = -1 if self.args.features == 'MS' else 0

                    # 计算损失
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())  # 将损失值添加到训练损失列表中

                # 每训练100个batch打印一次损失
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 计算每次迭代的时间和剩余时间
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 如果使用自动混合精度，则进行缩放后的反向传播和优化步骤
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 正常进行反向传播和优化步骤
                    loss.backward()
                    model_optim.step()

                # 如果学习率调整策略为'TST'，则调整学习率
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # 打印每个epoch的耗时
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 计算平均训练损失
            train_loss = np.average(train_loss)
            # 在验证集和测试集上进行评估
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 打印训练、验证和测试损失
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # 检查是否早停
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 如果学习率调整策略不是'TST'，则调整学习率
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # 加载最佳模型的权重
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 返回训练好的模型
        return self.model

    # 测试函数，用于评估模型在测试集上的表现
    def test(self, setting, test=0):
        # 获取测试数据和对应的加载器
        test_data, test_loader = self._get_data(flag='test')
        
        # 如果需要测试，则加载之前保存的模型权重
        if test:
            print('loading model')  # 输出加载模型的提示信息
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 定义检查点路径
        checkpoints_path = './checkpoints/' + setting + '/'
        # 初始化预测值和真实值的列表
        preds = []
        trues = []
        # 定义测试结果保存路径
        folder_path = './test_results/' + setting + '/'
        # 如果路径不存在，则创建路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 将模型设置为评估模式
        self.model.eval()
        # 禁用梯度计算，以节省内存和加速计算
        with torch.no_grad():
            # 遍历测试数据集的每个批次
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # 将输入数据和标签转换为浮点类型，并移动到设备（如GPU）上
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # 将时间标记数据转换为浮点类型并移动到设备上
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 如果数据集是'PEMS'或'Solar'，则不使用时间标记数据
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                # 如果不使用下采样层，则生成decoder的输入
                if self.args.down_sampling_layers == 0:
                    # 生成一个与batch_y最后n个时间步长相同形状的零张量，并与batch_y的前label_len个时间步长拼接
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # 编码器 - 解码器部分
                if self.args.use_amp:
                    # 如果使用自动混合精度进行前向传播
                    with torch.cuda.amp.autocast():
                        # 如果输出注意力图，则只取模型输出的第一个元素
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    # 正常进行前向传播
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 根据特征类型调整输出的维度
                f_dim = -1 if self.args.features == 'MS' else 0

                # 将输出和标签从GPU转移到CPU，并转换为NumPy数组
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # 将当前批次的预测值和真实值保存到列表中
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                # 每隔20个批次，进行一次可视化保存
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    # 如果需要反归一化数据，则进行处理
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    # 将输入、真实值和预测值连接起来，进行对比
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # 保存可视化图像
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 将预测值和真实值转换为NumPy数组
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)  # 输出预测值和真实值的形状
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)  # 输出重整后的形状

        # 如果数据集为'PEMS'，进行反归一化处理
        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 计算评估指标，如均方误差（MSE）、平均绝对误差（MAE）等
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))

        # 将结果写入文本文件
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        if self.args.data == 'PEMS':
            f.write('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))
        else:
            f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # 保存评估指标和预测值、真实值为NumPy数组
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return  # 返回结果


