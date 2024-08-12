# HVAC故障检测与预测系统

## 简介

本系统是一个基于[TimeMixer](https://github.com/kwuking/TimeMixer)开源深度学习算法的HVAC故障检测与预测工具。它利用Kafka、Spark、Hadoop等中间件高效采集数据，并通过特定的数据转换流程，将数据转化为brick格式进行处理和存储。系统能够实现对HVAC系统的实时监控和故障预测，帮助提高系统的可靠性和运维效率。

## 特点

- **实时监控**：系统能够实时采集HVAC系统数据。
- **故障预测**：基于深度学习算法，实现故障的早期预测。
- **数据转换**：将原始数据转换为brick格式，优化存储和处理效率。
- **开源算法**：使用清华大学开源的TimeMixer算法，保证算法的先进性和可靠性。

## 安装指南

1. 确保您的系统满足以下依赖：
   - Python 3.8 或更高版本
   - Kafka、Spark、Hadoop环境配置
3. 按照TimeMixer的README指南安装和配置环境。

## 快速开始

1. 启动Kafka、Spark、Hadoop服务。
2. 运行数据采集脚本，将HVAC系统数据采集并发送至Kafka。
3. 使用Spark处理数据，并转换为brick格式。
4. 运行故障预测脚本，加载TimeMixer模型进行预测。

## 配置选项

系统配置文件位于`config/settings.json`，您可以根据需要调整以下参数：

- `kafkaBrokers`: Kafka代理服务器地址。
- `topicName`: Kafka主题名称。
- `sparkMaster`: Spark集群的Master URL。
- `brickDataPath`: brick格式数据存储路径。

使用TimeMixer模型训练预测运行的示例代码位于`/text.py`，您可以根据需要调整相关参数运行。

模型参数文件位于`/cs.py`，有相关参数的注释说明。

## 示例代码

以下是使用系统进行故障预测的示例代码片段：

```python
# 假设您已经配置好环境并加载了TimeMixer模型
from timemixer import TimeMixerPredictor

# 初始化预测器
predictor = TimeMixerPredictor(model_path='path/to/timemixer/model')

# 加载brick格式数据
data = load_brick_data('path/to/brick/data')

# 进行故障预测
predictions = predictor.predict(data)

# 输出预测结果
print(predictions)
```

以下是使用TimeMixer模型训练预测的示例代码片段：

```python
# 假设您已经配置好环境并下载了本项目文件
import subprocess
import threading


def read_output(pipe, pipe_name):
    # 不断循环直到读取完所有数据
    while True:
        # 从管道中读取一行数据
        line = pipe.readline()
        # 如果没有数据（即读取到文件末尾），退出循环
        if not line:
            break
        # 打印读取到的行，去除首尾的空白字符
        print(line.strip())

# 要运行的Python文件路径
file_path = 'cs.py'

# 配置模型运行参数
parameters_brick_model__batch_1_1 = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', 'dataset/brick_model_1.3_merged/',
    '--data_path', 'brick_model_1.3_batch_1.ttl',
    '--model_id', 'brick_model_1.3_batch_1_96_96',
    '--model', 'TimeMixer',
    '--data', 'custom3',
    '--features', 'M',
    '--target', 'OaTemp',
    '--seq_len', '96',
    '--label_len', '0',
    '--pred_len', '96',
    '--e_layers', '4',
    '--enc_in', '20',
    '--dec_in', '20',
    '--c_out', '20',
    '--des', 'Exp',
    '--itr', '1',
    '--d_model', '16',
    '--d_ff', '32',
    '--batch_size', '16',
    '--learning_rate', '0.01',
    '--down_sampling_layers', '3',
    '--down_sampling_method', 'avg',
    '--down_sampling_window', '2',
    '--embed', 'learned',
    '--patience', '5',
    # '--channel_independence', '0',
    # '--freq', 't'
]

# 构建命令
command = ['python', file_path] + parameters_brick_model__batch_1_1

# 使用subprocess.Popen启动进程
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 创建线程来读取标准输出和标准错误流
stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "STDOUT"))
stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "STDERR"))

# 启动线程
stdout_thread.start()
stderr_thread.start()

# 等待进程完成
process.wait()

# 等待线程完成
stdout_thread.join()
stderr_thread.join()
```

## 许可证

本项目采用Apache License 2.0许可证。

## 致谢

感谢清华大学与蚂蚁集团开源的TimeMixer算法，以及所有为本项目做出贡献的开发者。

## 版本历史

- v1.0.0: 初始版本发布。

## 截图演示
