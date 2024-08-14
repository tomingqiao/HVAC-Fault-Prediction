from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider  # 导入数据提供者
from exp.exp_basic import Exp_Basic  # 导入基础实验类
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy  # 导入工具函数
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')  # 忽略警告信息

# 分类实验的类，继承自基础实验类 Exp_Basic
class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        # 初始化并继承自基类 Exp_Basic
        super(Exp_Classification, self).__init__(args)

    # 构建模型的函数
    def _build_model(self):
        # 根据数据集的最大序列长度和特征维度来设置模型参数
        train_data, train_loader = self._get_data(flag='TRAIN')  # 获取训练数据
        test_data, test_loader = self._get_data(flag='TEST')  # 获取测试数据
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)  # 设置序列长度
        self.args.pred_len = 0  # 预测长度设置为 0，因为这是分类任务
        self.args.enc_in = train_data.feature_df.shape[1]  # 设置输入特征的维度
        self.args.num_class = len(train_data.class_names)  # 设置类别数目
        
        # 初始化模型
        model = self.model_dict[self.args.model].Model(self.args).float()  # 构建模型并转换为浮点型
        if self.args.use_multi_gpu and self.args.use_gpu:  # 如果使用多GPU
            model = nn.DataParallel(model, device_ids=self.args.device_ids)  # 将模型并行化
        return model  # 返回构建的模型

    # 获取数据的函数
    def _get_data(self, flag):
        # 根据标志（TRAIN或TEST）获取对应的数据集和数据加载器
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # 选择优化器的函数
    def _select_optimizer(self):
        # 选择优化器，这里使用的是 RAdam 优化器
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # 选择损失函数的函数
    def _select_criterion(self):
        # 选择损失函数，这里使用的是交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        return criterion

    # 验证模型的函数
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []  # 用于存储总损失
        preds = []  # 用于存储预测结果
        trues = []  # 用于存储真实标签
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)  # 将数据转换为浮点型并移动到指定设备
                padding_mask = padding_mask.float().to(self.device)  # 移动 padding mask 到指定设备
                label = label.to(self.device)  # 移动标签到指定设备

                # 模型进行前向传播
                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()  # 获取预测结果
                loss = criterion(pred, label.long().squeeze())  # 计算损失
                total_loss.append(loss.item())  # 记录损失

                preds.append(outputs.detach())  # 记录预测结果
                trues.append(label)  # 记录真实标签

        total_loss = np.average(total_loss)  # 计算平均损失

        preds = torch.cat(preds, 0)  # 将预测结果拼接成一个向量
        trues = torch.cat(trues, 0)  # 将真实标签拼接成一个向量
        probs = torch.nn.functional.softmax(preds)  # 计算每个类别的概率
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # 获取预测的类别索引
        trues = trues.flatten().cpu().numpy()  # 将真实标签展平成一个向量
        accuracy = cal_accuracy(predictions, trues)  # 计算准确率

        self.model.train()  # 恢复模型为训练模式
        return total_loss, accuracy  # 返回验证损失和准确率

    # 训练模型的函数
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')  # 获取训练数据
        vali_data, vali_loader = self._get_data(flag='TEST')  # 获取验证数据
        test_data, test_loader = self._get_data(flag='TEST')  # 获取测试数据

        path = os.path.join(self.args.checkpoints, setting)  # 设置模型保存路径
        if not os.path.exists(path):  # 如果路径不存在，则创建
            os.makedirs(path)

        time_now = time.time()  # 获取当前时间

        train_steps = len(train_loader)  # 计算训练步骤数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 初始化早停机制

        model_optim = self._select_optimizer()  # 选择优化器
        criterion = self._select_criterion()  # 选择损失函数

        # 学习率调度器
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):  # 开始训练
            iter_count = 0  # 初始化迭代计数器
            train_loss = []  # 用于存储训练损失

            self.model.train()  # 设置模型为训练模式
            epoch_time = time.time()  # 记录 epoch 开始时间

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1  # 迭代计数加一
                model_optim.zero_grad()  # 清除梯度

                batch_x = batch_x.float().to(self.device)  # 将输入数据移动到设备上
                padding_mask = padding_mask.float().to(self.device)  # 移动 padding mask 到设备
                label = label.to(self.device)  # 移动标签到设备

                # 前向传播
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))  # 计算损失
                train_loss.append(loss.item())  # 记录损失

                if (i + 1) % 100 == 0:  # 每 100 次迭代打印一次信息
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count  # 计算每次迭代的时间
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)  # 估算剩余时间
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0  # 重置迭代计数器
                    time_now = time.time()  # 重置当前时间

                loss.backward()  # 反向传播
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)  # 梯度裁剪
                model_optim.step()  # 更新模型参数

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))  # 打印 epoch 消耗时间
            train_loss = np.average(train_loss)  # 计算平均训练损失
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)  # 验证模型
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)  # 测试模型

            # 打印损失和准确率
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            
            # 早停检查
            early_stopping(-test_accuracy, self.model, path)
            if early_stopping.early_stop:  # 如果早停条件满足
                print("Early stopping")  # 打印早停信息
                break  # 结束训练
            
            # 每 5 个 epoch 调整一次学习率
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)

        best_model_path = path + '/' + 'checkpoint.pth'  # 获取保存的最佳模型路径
        self.model.load_state_dict(torch.load(best_model_path))  # 加载最佳模型

        return self.model  # 返回模型

    # 测试模型的函数
    def test(self, setting):
        test_data, test_loader = self._get_data(flag='TEST')  # 获取测试数据

        # 如果存在多个测试数据集，则测试每一个数据集
        if isinstance(test_loader, list):
            for i, test_loader_ in enumerate(test_loader):
                self._test(test_data, test_loader_, setting, str(i))
        else:
            self._test(test_data, test_loader, setting)

    # 内部测试函数，进行实际的测试并输出结果
    def _test(self, test_data, test_loader, setting, str_postfix=''):
        self.model.eval()  # 设置模型为评估模式

        preds = []  # 用于存储预测结果
        trues = []  # 用于存储真实标签

        # 打开结果文件
        folder_path = './results/' + setting + '/'  # 结果文件夹路径
        if not os.path.exists(folder_path):  # 如果文件夹不存在则创建
            os.makedirs(folder_path)
        file_name = folder_path + 'real_prediction_result' + str_postfix + '.txt'  # 结果文件路径

        with torch.no_grad():  # 禁用梯度计算
            with open(file_name, 'w') as f:  # 打开文件
                for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)  # 将输入数据移动到设备上
                    padding_mask = padding_mask.float().to(self.device)  # 移动 padding mask 到设备
                    label = label.to(self.device)  # 移动标签到设备

                    outputs = self.model(batch_x, padding_mask, None, None)  # 前向传播

                    preds.append(outputs.detach())  # 记录预测结果
                    trues.append(label)  # 记录真实标签

                    # 将预测结果写入文件
                    probs = torch.nn.functional.softmax(outputs, dim=1)  # 计算每个类别的概率
                    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # 获取预测的类别索引
                    labels = label.flatten().cpu().numpy()  # 将真实标签展平成一个向量
                    for j in range(predictions.shape[0]):  # 对每个样本的预测结果
                        f.write(f'{predictions[j]}\t{labels[j]}\n')  # 写入预测类别和真实标签

        # 将预测结果和真实标签拼接
        preds = torch.cat(preds, 0).cpu().numpy()
        trues = torch.cat(trues, 0).cpu().numpy()

        # 计算分类结果
        result = classification_report(trues.flatten(), np.argmax(preds, axis=1), digits=4)
        print(result)  # 打印分类报告

        # 保存结果到文件
        f = open(folder_path + 'classification_report' + str_postfix + '.txt', 'w')
        f.write(result)  # 写入分类报告
        f.close()

