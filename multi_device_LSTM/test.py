import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_model import TimeSeriesDataset, LSTMModel
import os

def test_model(device_name):
    # 读取测试集的数据
    test_df = pd.read_csv(f'dataset/{device_name}_test_dataset.csv')

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型和归一化参数
    checkpoint = torch.load(f'{device_name}_best_model.pt', map_location=device)
    min_value = checkpoint['min_value']
    max_value = checkpoint['max_value']

    # 归一化测试数据
    test_data = (test_df['load_value'] - min_value) / (max_value - min_value)
    test_data = torch.tensor(test_data.values, dtype=torch.float32).unsqueeze(1)

    # 创建测试集的数据集对象
    seq_length = 10
    test_dataset = TimeSeriesDataset(test_data, seq_length)

    # 创建数据加载器
    batch_size = 32
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型参数
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1

    # 创建模型实例
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 加载已训练好的模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 在测试集上进行预测
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.detach().cpu().numpy())

    # 将预测结果转换为一维数组
    predictions = np.concatenate(predictions).flatten()

    # 反归一化
    predictions_original_scale = predictions * (max_value - min_value) + min_value

    # 计算 SMAPE (对称平均绝对百分比误差)
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

    epsilon = 1e-8  # 避免极小值问题
    true_values = test_df['load_value'].values[seq_length:]  # 排除前seq_length个值

    # 计算 SMAPE
    smape_value = smape(true_values, predictions_original_scale)
    print(f'Device: {device_name}, SMAPE: {smape_value:.2f}%')

    # 计算 MSE（均方误差）和 RMSE（均方根误差）
    mse = np.mean((true_values - predictions_original_scale) ** 2)
    rmse = np.sqrt(mse)

    # 计算真实值的平均值
    mean_true_value = np.mean(true_values)

    # 将 MSE 和 RMSE 转换为百分比
    mse_percentage = (mse / (mean_true_value ** 2)) * 100
    rmse_percentage = (rmse / mean_true_value) * 100

    print(f'Device: {device_name}, MSE: {mse_percentage:.2f}%')
    print(f'Device: {device_name}, RMSE: {rmse_percentage:.2f}%')

    # 绘制预测值和真实值的曲线图
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index[seq_length:], true_values, label='True Values')
    plt.plot(test_df.index[seq_length:], predictions_original_scale, label='Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Load_Value')
    plt.title(f'Predicted vs True Values for {device_name}')
    plt.legend()
    plt.savefig(f'{device_name}_prediction_plot.png')
    plt.close()

# 主程序
if __name__ == "__main__":
    # 获取dataset目录下的所有CSV文件
    dataset_dir = 'dataset'
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('_test_dataset.csv')]

    # 提取设备名并测试每个设备的模型
    for csv_file in csv_files:
        device_name = csv_file.replace('_test_dataset.csv', '')
        print(f"Testing model for device: {device_name}")
        test_model(device_name)