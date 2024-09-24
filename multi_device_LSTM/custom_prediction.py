import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_model import TimeSeriesDataset, LSTMModel
import os
from datetime import datetime, timedelta

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

def predict_custom_range(device_name, start_date, end_date):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型和归一化参数
    checkpoint = torch.load(f'{device_name}_best_model.pt', map_location=device)
    min_value = checkpoint['min_value']
    max_value = checkpoint['max_value']

    # 创建模型实例
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 加载已训练好的模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 读取最后一个已知的数据点
    train_df = pd.read_csv(f'dataset/{device_name}_train_dataset.csv')
    last_known_data = train_df['load_value'].values[-10:]  # 假设我们需要10个历史数据点

    # 归一化最后已知的数据
    last_known_data_normalized = (last_known_data - min_value) / (max_value - min_value)
    last_known_data_normalized = torch.tensor(last_known_data_normalized, dtype=torch.float32).unsqueeze(1).to(device)

    # 生成预测日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    predictions = []

    with torch.no_grad():
        for _ in range(len(date_range)):
            input_seq = last_known_data_normalized[-10:].unsqueeze(0)  # 保持输入序列为10
            output = model(input_seq)
            predictions.append(output.item())
            last_known_data_normalized = torch.cat((last_known_data_normalized[1:], output), 0)

    # 反归一化预测结果
    predictions_original_scale = np.array(predictions) * (max_value - min_value) + min_value

    # 创建包含日期和预测值的DataFrame
    results_df = pd.DataFrame({
        'Date': date_range,
        'Predicted_Load': predictions_original_scale
    })

    # 保存预测结果
    results_df.to_csv(f'{device_name}_custom_predictions.csv', index=False)

    # 绘制预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Date'], results_df['Predicted_Load'])
    plt.xlabel('Date')
    plt.ylabel('Predicted Load')
    plt.title(f'Load Prediction for {device_name} ({start_date} to {end_date})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{device_name}_custom_prediction_plot.png')
    plt.close()

    print(f"Custom predictions for {device_name} from {start_date} to {end_date} have been saved.")

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

        # 进行自定义时间范围的预测
        start_date = '2023-01-01'  # 设置你想要预测的起始日期
        end_date = '2024-12-31'    # 设置你想要预测的结束日期
        predict_custom_range(device_name, start_date, end_date)