import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from train_model import LSTMPredictor
import argparse

def load_data(data_dir):
    all_data = pd.DataFrame()
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            csv_files.sort(key=lambda x: int(x.split('.')[0][3:]))
            
            for file in csv_files:
                file_path = os.path.join(folder_path, file)
                data = pd.read_csv(file_path)
                all_data = pd.concat([all_data, data])

    all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
    all_data = all_data.sort_values(by='timestamp')
    all_data = all_data.set_index('timestamp')
    return all_data

def predict_future_load(location_name, all_data, future_days, seq_length):
    model = LSTMPredictor()
    model.load_state_dict(torch.load(f'lstm_model_{location_name}.pth'))
    model.eval()
    scaler = torch.load(f'scaler_{location_name}.pth')

    location_data = all_data[all_data['location'] == location_name]['load_value'].values
    
    input_sequence = location_data[-seq_length:]
    scaled_input = scaler.transform(input_sequence.reshape(-1, 1))

    predictions = []
    test_inputs = scaled_input.tolist()  # 使用 scaled_input 的列表表示形式

    for _ in range(future_days * 96):  # 每天96个15分钟间隔
        # 准备输入张量，确保维度为 (1, seq_length, 1)
        seq = torch.FloatTensor(test_inputs[-seq_length:]).unsqueeze(0)  # 形状为 (1, seq_length, 1)
        
        with torch.no_grad():
            hidden_cell = model.init_hidden(seq.size(0))  # batch size = 1
            
            # 直接使用 seq，而不是再增加维度
            y_pred, hidden_cell = model(seq, hidden_cell) 
        
        # 将预测结果添加到 test_inputs 中，用于未来预测
        test_inputs.append([y_pred.item()])  # 确保格式一致

    # 将预测结果转换回原始尺度
    predicted_load = scaler.inverse_transform(np.array(test_inputs[-future_days*96:]).reshape(-1, 1))
    return predicted_load

def create_prediction_dataframe(location_name, predictions, start_date):
    """
    创建包含预测结果的DataFrame
    """
    timestamps = [start_date + timedelta(minutes=15*i) for i in range(len(predictions))]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'location': location_name,
        'load_value': predictions.flatten()
    })
    return df

def export_predictions_to_csv(predictions_df, output_file):
    """
    将预测结果导出为CSV文件
    """
    predictions_df.to_csv(output_file, index=False)
    print(f"预测结果已导出到 {output_file}")

def get_file_name(prefix, location_name, start_date, end_date, current_date):
    """
    生成包含预测时间段和当前日期的文件名
    """
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    current_str = current_date.strftime("%Y%m%d")
    return f"{prefix}_{location_name}_{start_str}_to_{end_str}_created_{current_str}.csv"

if __name__ == "__main__":
    
    # 定义默认值
    default_data_dir = './data/'
    default_seq_length = 30
    default_future_days = 7

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Predict future load values")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Directory containing the data files")
    parser.add_argument("--seq_length", type=int, default=default_seq_length, help="Sequence length for LSTM input")
    parser.add_argument("--future_days", type=int, default=default_future_days, help="Number of days to predict into the future")
    args = parser.parse_args()

    # 使用解析的参数或默认值
    data_dir = args.data_dir
    seq_length = args.seq_length
    future_days = args.future_days

    print(f"Using parameters: data_dir={data_dir}, seq_length={seq_length}, future_days={future_days}")

    all_data = load_data(data_dir)

    locations = all_data['location'].unique()
    
    # 创建输出目录（如果不存在）
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)

    all_predictions = []

    # 获取预测的起始日期和结束日期
    start_date = all_data.index[-1] + timedelta(minutes=15)
    end_date = start_date + timedelta(days=future_days) - timedelta(minutes=15)
    current_date = datetime.now()

    for location_name in locations:
        predictions = predict_future_load(location_name, all_data, future_days, seq_length)
        
        predictions_df = create_prediction_dataframe(location_name, predictions, start_date)
        all_predictions.append(predictions_df)
        
        # 为每个位置创建单独的CSV文件
        location_output_file = os.path.join(output_dir, get_file_name("predictions", location_name, start_date, end_date, current_date))
        export_predictions_to_csv(predictions_df, location_output_file)
        
        print(f"未来 {future_days} 天 {location_name} 的预测负载:")
        print(predictions_df.head())

    # 将所有预测结果合并为一个DataFrame
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # 按时间戳和位置对合并后的预测结果进行排序
    combined_predictions = combined_predictions.sort_values(['timestamp', 'location'])

    # 将合并后的预测结果导出为CSV文件
    combined_output_file = os.path.join(output_dir, get_file_name("predictions_combined", "all_locations", start_date, end_date, current_date))
    export_predictions_to_csv(combined_predictions, combined_output_file)

    print("所有位置的预测结果已分别导出完成，并且已生成合并的预测结果文件。")