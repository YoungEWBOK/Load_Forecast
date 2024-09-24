import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# 数据读取与预处理
data_dir = './data/'
all_data = pd.DataFrame()

import os
import pandas as pd

# 初始化一个空的 DataFrame 来存储所有数据
all_data = pd.DataFrame()

# 定义一个函数来从文件名中提取日期
def extract_date(filename):
    # 假设文件名格式为 "day1.csv", "day2.csv" 等
    return int(filename.split('.')[0][3:])

# 读取所有 CSV 文件的数据
for folder in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        # 获取文件夹中的所有 CSV 文件并按日期排序
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        csv_files.sort(key=extract_date)
        
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            
            # 打印读取的文件名和数据的前几行
            print(f'Read data from: {file_path}')
            print(data.head())  # 显示前5行数据
            
            all_data = pd.concat([all_data, data])

# 确保按时间戳排序
all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])  # 转换为日期时间格式
all_data = all_data.sort_values(by='timestamp')  # 按时间戳升序排列

# 最后显示合并后的数据的前几行
print('All data after merging and sorting:')
print(all_data.head())

# 假设数据文件有三列: 'timestamp', 'location' 和 'load_value'
all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
all_data = all_data.set_index('timestamp')

# 按 location 分组数据
location_data = {}
for location in all_data['location'].unique():
    location_data[location] = all_data[all_data['location'] == location].sort_values(by='timestamp')

print("Data separated by location:")
for location, data in location_data.items():
    print(f"\n{location}:")
    print(data.reset_index())

# 创建 PyTorch 数据集
class LoadDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        seq_x = self.data[idx:idx+self.seq_length]
        seq_y = self.data[idx+self.seq_length]
        return torch.Tensor(seq_x), torch.Tensor(seq_y)

# 定义 LSTM 模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq, hidden_cell):
        lstm_out, hidden_cell = self.lstm(input_seq, hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions, hidden_cell

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_layer_size),
                torch.zeros(1, batch_size, self.hidden_layer_size))

# 训练模型并保存权重
def train_model(location_data, location_name, seq_length=30, epochs=20):
    # 提取负荷数据
    load_data = location_data['load_value'].values.reshape(-1, 1)
    
    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(load_data)

    # 创建数据集和数据加载器
    dataset = LoadDataset(scaled_data, seq_length)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 构建 LSTM 模型
    model = LSTMPredictor()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            hidden_cell = model.init_hidden(seq.size(0))
            y_pred, hidden_cell = model(seq, hidden_cell)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        
        if epoch % 2 == 0:
            print(f'Location: {location_name}, Epoch {epoch} Loss: {single_loss.item()}')

    # 保存模型权重
    torch.save(model.state_dict(), f'lstm_model_{location_name}.pth')
    
    # 保存 scaler
    torch.save(scaler, f'scaler_{location_name}.pth')

    print(f"Model for location {location_name} trained and saved successfully.")

# 运行训练
if __name__ == "__main__":
    seq_length = 30
    epochs = 20
    
    for location, data in location_data.items():
        print(f"\nTraining model for {location}")
        train_model(data, location, seq_length, epochs)