import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_file, output_dir, test_size=0.1, val_size=0.2):
    # 读取处理后的数据集
    df = pd.read_csv(input_file)

    # 将时间列转换为日期时间格式
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    # 将负荷值列转换为浮点数
    df['load_value'] = df['load_value'].astype(float)

    # 按升序对数据进行排序
    df.sort_values('timestamp', ascending=True, inplace=True)

    # 计算训练集比例
    train_ratio = 1 - test_size - val_size

    # 将数据集划分为训练集、验证集和测试集
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=val_size/(train_ratio+val_size), shuffle=False)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存划分后的数据集为CSV文件
    train_data.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, 'val_dataset.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)
    print('数据划分已完成')

if __name__ == "__main__":
    # 如果直接运行这个脚本，使用默认参数
    split_dataset('./dataset/origin_dataset.csv', './dataset')