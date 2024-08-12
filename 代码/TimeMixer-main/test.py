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
model_name = 'TimeMixer'

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16

ETTm1_parameters = [
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--root_path", "./dataset/ETT-small/",
    "--data_path", "ETTm1.csv",
    "--model_id", f"ETTm1_{seq_len}_336",
    "--model", model_name,
    "--data", "ETTm1",
    "--features", "M",
    "--seq_len", str(seq_len),
    "--label_len", "0",
    "--pred_len", "336",
    "--e_layers", str(e_layers),
    "--enc_in", "7",
    "--c_out", "7",
    "--des", "Exp",
    "--itr", "1",
    "--d_model", str(d_model),
    "--d_ff", str(d_ff),
    "--batch_size", str(batch_size),
    "--learning_rate", str(learning_rate),
    "--down_sampling_layers", str(down_sampling_layers),
    "--down_sampling_method", "avg",
    "--down_sampling_window", str(down_sampling_window)
]


parameters_brick_model__batch_1 = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', 'dataset/brick_model_1.3_merged/',
    '--data_path', 'brick_model_1.3_batch_1.ttl',
    '--model_id', 'brick_model_1.3_batch_1_6_96_96',
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
    '--d_model', '32',
    '--d_ff', '32',
    '--batch_size', '16',
    '--learning_rate', '0.01',
    '--down_sampling_layers', '5',
    '--down_sampling_method', 'avg',
    '--down_sampling_window', '2',
    '--embed', 'learned',
    # '--channel_independence', '0',
    # '--freq', 't'
]

parameters_brick_model__batch_3 = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', 'dataset/brick_model_1.3_merged/',
    '--data_path', 'brick_model_1.3_batch_3.ttl',
    '--model_id', 'brick_model_1.3_batch_3_96_96',
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

parameters_brick_model__batch_4 = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', 'dataset/brick_model_1.3_merged/',
    '--data_path', 'brick_model_1.3_batch_4.ttl',
    '--model_id', 'brick_model_1.3_batch_4_96_96',
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

parameters_brick_model__batch_4_text = [
    '--task_name', 'long_term_forecast',
    '--is_training', '0',
    '--root_path', 'dataset/brick_model_1.3_merged/',
    '--data_path', 'brick_model_1.3_batch_4.ttl',
    '--model_id', 'brick_model_1.3_batch_4_96_96',
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

parameters_Three_Years_Daily_Data = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', 'dataset/Five_Years_Daily_Data/',
    '--data_path', 'Three_Years_Daily_Data.ttl',
    '--model_id', 'Three_Years_Daily_Data_96_96',
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
