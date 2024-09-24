import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(device_name, df):
    # 将数据集划分为训练集、验证集和测试集
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    train_data, test_data = train_test_split(df, test_size=test_ratio, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=val_ratio/(train_ratio+val_ratio), shuffle=False)

    # 保存划分后的数据集为CSV文件
    train_data.to_csv(f'./dataset/{device_name}_train_dataset.csv', index=False)
    val_data.to_csv(f'./dataset/{device_name}_val_dataset.csv', index=False)
    test_data.to_csv(f'./dataset/{device_name}_test_dataset.csv', index=False)
    print(f'数据划分已完成: {device_name}')

# 主程序
if __name__ == "__main__":
    # 获取所有设备的CSV文件
    dataset_dir = './dataset'
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('_origin_dataset.csv')]

    for csv_file in csv_files:
        # 从文件名中提取设备名
        device_name = csv_file.replace('_origin_dataset.csv', '')
        
        # 读取处理后的数据集
        df = pd.read_csv(os.path.join(dataset_dir, csv_file))
        
        # 对每个设备的数据集进行划分
        split_dataset(device_name, df)