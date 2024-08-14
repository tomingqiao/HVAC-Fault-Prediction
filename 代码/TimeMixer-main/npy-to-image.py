import numpy as np  # 导入NumPy库，用于处理数组和数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘制图形

# 加载预测值和真实值文件
filename = "long_term_forecast_brick_model_1.3_batch_1_96_96_none_TimeMixer_custom3_sl96_pl96_dm16_nh4_el4_dl1_df32_fc1_eblearned_dtTrue_Exp_0"

# 从指定路径加载预测值的NumPy数组
pred = np.load("./results" + "/" + filename + "/pred.npy")
# 从指定路径加载真实值的NumPy数组
true = np.load("./results" + "/" + filename + "/true.npy")

# 提取需要绘制的数据：取出真实值和预测值的最后一列数据
true_last_column = true[0, :, -1]  # 获取真实值的最后一列数据
pred_last_column = pred[0, :, -1]  # 获取预测值的最后一列数据

# # 组合成一个整体的数组（此部分代码被注释掉）
# gt = np.concatenate((true_last_column, true_last_column), axis=0)  # 将真实值的最后一列数据与自身拼接
# pd = np.concatenate((true_last_column, pred_last_column), axis=0)  # 将真实值的最后一列数据与预测值拼接

# 绘制真实值和预测值的数据曲线
plt.figure(figsize=(10, 6))  # 创建一个大小为10x6的图形
plt.plot(true_last_column, label='Ground Truth')  # 绘制真实值曲线，标签为“Ground Truth”
plt.plot(pred_last_column, label='Predicted')  # 绘制预测值曲线，标签为“Predicted”
plt.title('Comparison of Ground Truth and Predicted Data')  # 设置图形标题
plt.xlabel('Time')  # 设置X轴标签为“Time”
plt.ylabel('Value')  # 设置Y轴标签为“Value”
plt.legend()  # 显示图例，区分不同的曲线
plt.grid(True)  # 启用网格显示
plt.show()  # 显示绘制的图形
