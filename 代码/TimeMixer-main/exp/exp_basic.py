import os
import torch
from models import TimeMixer  # 导入模型模块

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args  # 保存传入的参数
        self.model_dict = {
            'TimeMixer': TimeMixer,  # 模型字典，用于映射模型名称到实际模型类
        }
        self.device = self._acquire_device()  # 获取计算设备（CPU或GPU）
        self.model = self._build_model().to(self.device)  # 构建模型并将其移动到指定设备上

    def _build_model(self):
        raise NotImplementedError  # 抛出未实现异常，表示子类需要重写此方法
        return None  # 返回空值（仅为占位，实际不会执行到此行）

    def _acquire_device(self):
        # 判断是否使用GPU
        if self.args.use_gpu:
            import platform  # 导入平台模块，用于检测操作系统
            if platform.system() == 'Darwin':  # 检查是否为 macOS 系统
                device = torch.device('mps')  # 如果是，使用 MPS 设备
                print('Use MPS')  # 打印提示信息
                return device  # 返回 MPS 设备
            # 设置可见的 CUDA 设备
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))  # 选择指定的 CUDA 设备
            if self.args.use_multi_gpu:
                print('Use GPU: cuda{}'.format(self.args.device_ids))  # 打印多GPU提示信息
            else:
                print('Use GPU: cuda:{}'.format(self.args.gpu))  # 打印单GPU提示信息
        else:
            device = torch.device('cpu')  # 如果不使用GPU，则使用CPU
            print('Use CPU')  # 打印提示信息
        return device  # 返回选择的设备

    def _get_data(self):
        pass  # 获取数据的方法，留作将来实现

    def vali(self):
        pass  # 验证模型的方法，留作将来实现

    def train(self):
        pass  # 训练模型的方法，留作将来实现

    def test(self):
        pass  # 测试模型的方法，留作将来实现
