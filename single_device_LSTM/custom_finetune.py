import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_model import TimeSeriesDataset, LSTMModel
from datetime import datetime, timedelta

def load_model_and_params(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    min_value = checkpoint['min_value']
    max_value = checkpoint['max_value']

    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, min_value, max_value, device

def predict_future(start_date, num_days, model_path='best_model.pt', history_days=30):
    model, min_value, max_value, device = load_model_and_params(model_path)

    # 读取训练数据的最后 history_days 天作为初始序列
    train_df = pd.read_csv('./dataset/train_dataset.csv')
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    last_date = train_df['timestamp'].max().date()
    start_date_history = last_date - timedelta(days=history_days)
    history_data = train_df[train_df['timestamp'].dt.date >= start_date_history]['load_value'].values

    # 归一化历史数据
    history_data_normalized = (history_data - min_value) / (max_value - min_value)
    history_data_normalized = torch.tensor(history_data_normalized, dtype=torch.float32).unsqueeze(1).to(device)

    # 生成预测日期范围
    start_datetime = pd.to_datetime(start_date)
    end_datetime = start_datetime + timedelta(days=num_days) - timedelta(minutes=15)
    date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='15min')

    predictions = []
    seq_length = 96 * 7  # 使用过去7天的数据进行预测

    with torch.no_grad():
        for _ in range(len(date_range)):
            input_seq = history_data_normalized[-seq_length:].unsqueeze(0)
            output = model(input_seq)
            predictions.append(output.item())
            history_data_normalized = torch.cat((history_data_normalized[1:], output), 0)

    # 反归一化预测结果
    predictions_original_scale = np.array(predictions) * (max_value - min_value) + min_value

    results_df = pd.DataFrame({
        'timestamp': date_range,
        'load_value': predictions_original_scale
    })

    return results_df

def save_results(df, filename):
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def plot_results(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['load_value'])
    plt.xlabel('Date')
    plt.ylabel('Load Value')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

def calculate_statistics(df):
    daily_stats = df.resample('D', on='timestamp').agg({
        'load_value': ['mean', 'max']
    }).reset_index()
    daily_stats.columns = ['date', 'daily_mean', 'daily_max']
    return daily_stats

if __name__ == "__main__":
    start_date = '2023-09-01 00:00:00'  # 假设最后的数据点是 2023-08-31 23:45:00

    # 预测未来一天、三天、一个月和三个月
    prediction_periods = [1, 3, 30, 90]

    for days in prediction_periods:
        results = predict_future(start_date, days, history_days=30)
        
        csv_filename = f'predictions_{days}days.csv'
        save_results(results, csv_filename)
        
        plot_filename = f'predictions_{days}days_plot.png'
        plot_title = f'Load Prediction for Next {days} Days'
        plot_results(results, plot_title, plot_filename)

        # 计算并保存每日统计数据
        daily_stats = calculate_statistics(results)
        stats_filename = f'daily_stats_{days}days.csv'
        save_results(daily_stats, stats_filename)

    print("All predictions completed.")