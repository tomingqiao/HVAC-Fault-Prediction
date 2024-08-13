import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np

def summary(model, input_size, batch_size=-1, device="cuda"):
    # 注册hook函数以获取每层的输入输出信息
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]  # 获取模块类名
            module_idx = len(summary)  # 当前模块的索引

            m_key = "%s-%i" % (class_name, module_idx + 1)  # 为模块生成唯一键
            summary[m_key] = OrderedDict()  # 为模块创建字典条目
            
            # 获取输入的形状
            if isinstance(input[0], (list, tuple)):
                summary[m_key]["input_shape"] = [
                    [-1] + list(i.size())[1:] for i in input[0]
                ]
                summary[m_key]["input_shape"][0] = batch_size  # 设置批量大小
            else:
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size

            # 获取输出的形状
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            # 计算参数数量
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))  # 乘积获取权重参数数量
                summary[m_key]["trainable"] = module.weight.requires_grad  # 是否可训练
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))  # 乘积获取偏置参数数量
            summary[m_key]["nb_params"] = params  # 保存参数数量

        # 对于非Sequential、非ModuleList且非顶层模型的模块，注册hook
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # 确保设备为cuda或cpu
    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    # 根据设备选择数据类型
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # 支持多个输入
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # 使用大小为2的batch来处理batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # 创建属性
    summary = OrderedDict()  # 存储每层的摘要信息
    hooks = []  # 存储所有注册的hook

    # 注册hook
    model.apply(register_hook)

    # 前向传播
    model(*x)

    # 移除hook
    for h in hooks:
        h.remove()

    # 打印结果摘要
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0  # 总参数量
    total_output = 0  # 总输出大小
    trainable_params = 0  # 可训练参数量
    for layer in summary:
        # 打印每一层的输出形状和参数数量
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # 估算模型大小
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # 输出大小（MB），x2因考虑梯度
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))  # 参数大小（MB）

    # 打印整体统计信息
    print("================================================================")
    print("Total params: {0:,}".format(total_params))  # 总参数量
    print("Trainable params: {0:,}".format(trainable_params))  # 可训练参数量
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))  # 不可训练参数量
    print("----------------------------------------------------------------")
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)  # 前向/反向传递的大小
    print("Params size (MB): %0.2f" % total_params_size)  # 参数大小
    print("----------------------------------------------------------------")
