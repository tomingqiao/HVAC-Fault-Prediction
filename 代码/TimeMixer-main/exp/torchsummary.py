import torch  # 导入PyTorch库，用于深度学习操作
import torch.nn as nn  # 导入神经网络模块
from torch.autograd import Variable  # 导入自动求导工具

from collections import OrderedDict  # 导入有序字典，用于存储模型摘要信息
import numpy as np  # 导入NumPy库，用于数值计算


def summary(model, input_size, batch_size=-1, device="cuda"):  # 定义summary函数，用于生成模型的摘要信息
    # 内部函数：注册钩子函数，用于捕获模块的输入输出信息
    def register_hook(module):

        # 钩子函数：获取模块的输入输出信息
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]  # 获取模块的类名
            module_idx = len(summary)  # 模块索引，根据当前summary字典的长度确定

            m_key = "%s-%i" % (class_name, module_idx + 1)  # 构建模块的键名，格式为'类名-索引'
            summary[m_key] = OrderedDict()  # 在summary中创建一个新的有序字典条目

            # 如果输入是列表或元组，则获取每个输入的形状
            if isinstance(input[0], (list, tuple)):
                summary[m_key]["input_shape"] = [
                    [-1] + list(i.size())[1:] for i in input[0]
                ]
                summary[m_key]["input_shape"][0] = batch_size  # 设置批量大小为指定值
            else:
                summary[m_key]["input_shape"] = list(input[0].size())  # 获取输入的形状
                summary[m_key]["input_shape"][0] = batch_size  # 设置批量大小为指定值

            # 如果输出是列表或元组，则获取每个输出的形状
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())  # 获取输出的形状
                summary[m_key]["output_shape"][0] = batch_size  # 设置批量大小为指定值

            # 初始化参数数量
            params = 0
            # 如果模块有权重，则计算权重的参数数量
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad  # 判断权重是否需要梯度
            # 如果模块有偏置，则计算偏置的参数数量
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params  # 将参数数量记录在summary中

        # 过滤掉Sequential和ModuleList类型，以及模型自身，避免重复注册钩子
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))  # 注册前向钩子

    device = device.lower()  # 将设备名称转换为小写
    # 确保设备名称有效，只能是'cuda'或'cpu'
    assert device in [
        "cuda",
        "cpu",
    ], "输入设备无效，请指定'cuda'或'cpu'"

    # 设置数据类型，根据设备选择对应的浮点数类型
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # 如果输入尺寸是一个元组，则将其包装为列表
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # 为了批量归一化，设置批量大小为2
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]  # 创建随机输入张量
    # print(type(x[0]))

    # 创建存储属性的字典
    summary = OrderedDict()
    hooks = []  # 初始化钩子列表

    # 为模型的每个模块注册钩子
    model.apply(register_hook)

    # 执行前向传递
    # print(x.shape)
    model(*x)

    # 移除所有钩子
    for h in hooks:
        h.remove()

    # 打印模型摘要的表头
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    
    total_params = 0  # 总参数数量
    total_output = 0  # 总输出大小
    trainable_params = 0  # 可训练的参数数量
    
    # 遍历summary字典，打印每一层的摘要信息
    for layer in summary:
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]  # 累计参数数量
        total_output += np.prod(summary[layer]["output_shape"])  # 累计输出大小
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]  # 累计可训练参数数量
        print(line_new)  # 打印当前层的摘要信息

    # 假设每个数字占用4字节（在CUDA上为浮点数）
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2是为了考虑梯度
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))  # 计算参数大小（MB）

    # 打印模型的总体摘要信息
    print("================================================================")
    print("Total params: {0:,}".format(total_params))  # 打印总参数数量
    print("Trainable params: {0:,}".format(trainable_params))  # 打印可训练参数数量
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))  # 打印不可训练参数数量
    print("----------------------------------------------------------------")
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)  # 打印前向/反向传递大小
    print("Params size (MB): %0.2f" % total_params_size)  # 打印参数大小
    print("----------------------------------------------------------------")
    # return summary  # 返回summary字典（可选）
