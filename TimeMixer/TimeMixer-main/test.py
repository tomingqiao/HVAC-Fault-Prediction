import subprocess
import threading


def read_output(pipe, pipe_name):
    while True:
        line = pipe.readline()
        if not line:
            break
        print(line.strip())


model_name = 'TimeMixer'

seq_len = 96
e_layers = 3
down_sampling_layers = 3
down_sampling_window = 2
learning_rate = 0.01
d_model = 16
d_ff = 32
batch_size = 32
train_epochs = 20
patience = 10


# 要运行的Python文件路径
file_path = 'cs.py'

# 参数列表
parameters = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', './dataset/Data_Article_Dataset/',
    '--data_path', 'Data_Article_Dataset.csv',
    '--model_id', f'Data_Article_Dataset_{seq_len}_96',
    '--model', model_name,
    '--data', 'custom',
    '--features', 'M',
    '--target', 'ZoneTemp_2',
    '--freq', 't',
    '--seq_len', str(seq_len),
    '--label_len', '0',
    '--pred_len', '96',
    '--e_layers', str(e_layers),
    '--d_layers', '1',
    '--factor', '3',
    '--enc_in', '20',
    '--dec_in', '20',
    '--c_out', '20',
    '--des', 'Exp',
    '--itr', '1',
    '--d_model', str(d_model),
    '--d_ff', str(d_ff),
    '--batch_size', str(batch_size),
    '--learning_rate', str(learning_rate),
    '--train_epochs', str(train_epochs),
    '--patience', str(patience),
    '--down_sampling_layers', str(down_sampling_layers),
    '--down_sampling_method', 'avg',
    '--down_sampling_window', str(down_sampling_window)
]

parameters2 = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', './dataset/ETT-small/',
    '--data_path', 'ETTm1.csv',
    '--model_id', "ETTm1_96'_'96",
    '--model', 'TimeMixer',
    '--data', 'ETTm1',
    '--features', 'M',
    '--seq_len', '96',
    '--label_len', '0',
    '--pred_len', '96',
    '--e_layers', '2',
    '--enc_in', '7',
    '--c_out', '7',
    '--des', 'Exp',
    '--itr', '1',
    '--d_model', '16',
    '--d_ff', '32',
    '--batch_size', '16',
    '--learning_rate', '0.01',
    '--down_sampling_layers', '3',
    '--down_sampling_method', 'avg',
    '--down_sampling_window', '2'
]

parameters3 = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', './dataset/brick_model_1.3_merged/',
    '--data_path', 'brick_model_1.3_merged.ttl',
    '--model_id', f'brick_model_1.3_merged_{seq_len}_96',
    '--model', model_name,
    '--data', 'custom2',
    '--features', 'M',
    '--target', 'DaTemp',
    '--freq', 't',
    '--seq_len', str(seq_len),
    '--label_len', '0',
    '--pred_len', '96',
    '--e_layers', str(e_layers),
    '--d_layers', '1',
    '--factor', '3',
    '--enc_in', '20',
    '--dec_in', '20',
    '--c_out', '20',
    '--des', 'Exp',
    '--itr', '1',
    '--d_model', str(d_model),
    '--d_ff', str(d_ff),
    '--batch_size', str(batch_size),
    '--learning_rate', str(learning_rate),
    '--train_epochs', str(train_epochs),
    '--patience', str(patience),
    '--down_sampling_layers', str(down_sampling_layers),
    '--down_sampling_method', 'avg',
    '--down_sampling_window', str(down_sampling_window)
]

# 构建命令
command = ['python', file_path] + parameters3

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
