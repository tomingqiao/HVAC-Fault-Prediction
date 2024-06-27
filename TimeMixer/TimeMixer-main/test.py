import subprocess
import threading


def read_output(pipe, pipe_name):
    while True:
        line = pipe.readline()
        if not line:
            break
        print(line.strip())


# 要运行的Python文件路径
file_path = 'cs.py'

# 参数列表
parameters = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', './dataset/Data_Article_Dataset/',
    '--data_path', 'Data_Article_Dataset.csv',
    '--model_id', 'Data_Article_Dataset_96_96',
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

# 构建命令
command = ['python', file_path] + parameters

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
