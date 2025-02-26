import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset_model import TimeSeriesDataset, LSTMModel
import os

def train_model(device_name):
    # 读取训练集和验证集的数据
    train_df = pd.read_csv(f'dataset/{device_name}_train_dataset.csv')
    val_df = pd.read_csv(f'dataset/{device_name}_val_dataset.csv')

    # 记录最小值和最大值用于归一化
    min_value = train_df['load_value'].min()
    max_value = train_df['load_value'].max()

    # 归一化数据
    train_df['load_value'] = (train_df['load_value'] - min_value) / (max_value - min_value)
    val_df['load_value'] = (val_df['load_value'] - min_value) / (max_value - min_value)

    # 将数据集转换为PyTorch的Tensor
    train_data = torch.tensor(train_df['load_value'].values, dtype=torch.float32).unsqueeze(1)
    val_data = torch.tensor(val_df['load_value'].values, dtype=torch.float32).unsqueeze(1)

    # 创建训练集和验证集的数据集对象
    seq_length = 10
    train_dataset = TimeSeriesDataset(train_data, seq_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length)

    # 创建数据加载器
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型参数
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1

    # 创建模型实例
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 设置训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将模型移动到训练设备
    model.to(device)

    # 定义训练参数
    num_epochs = 500
    learning_rate = 0.001

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化最好的验证集损失
    best_val_loss = float('inf')

    # 训练模型
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        train_loss = 0.0

        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # 在验证集上进行评估
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

            val_loss /= len(val_dataloader)

        # 打印训练结果
        print(f'Device: {device_name}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 保存最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'min_value': min_value,
                'max_value': max_value
            }, f'{device_name}_best_model.pt')

    print(f"Training completed for device: {device_name}")

# 主程序
if __name__ == "__main__":
    # 获取dataset目录下的所有CSV文件
    dataset_dir = 'dataset'
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('_train_dataset.csv')]

    # 提取设备名并训练每个设备的模型
    for csv_file in csv_files:
        device_name = csv_file.replace('_train_dataset.csv', '')
        print(f"Training model for device: {device_name}")
        train_model(device_name)