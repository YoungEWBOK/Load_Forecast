import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 创建随机负荷数据的函数
def generate_load_data(start_date, end_date, locations, save_dir):
    current_date = start_date
    while current_date <= end_date:
        # 生成每天的数据
        date_str = current_date.strftime('%Y-%m-%d')
        folder_name = current_date.strftime('%Y-%m')
        file_name = f"day{current_date.day}.csv"
        
        # 创建目录
        folder_path = os.path.join(save_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # 每天的数据从 00:00 到 23:45，每15分钟一条
        timestamps = [current_date + timedelta(minutes=15*i) for i in range(96)]
        
        # 每个地点生成随机负荷值
        data = []
        for location in locations:
            load_values = np.random.uniform(low=100, high=150, size=len(timestamps))  # 生成负荷值
            for ts, load in zip(timestamps, load_values):
                data.append([ts.strftime('%Y-%m-%d %H:%M:%S'), location, load])
        
        # 保存为CSV文件
        df = pd.DataFrame(data, columns=["timestamp", "location", "load_value"])
        file_path = os.path.join(folder_path, file_name)
        df.to_csv(file_path, index=False)
        
        # 下一个日期
        current_date += timedelta(days=1)

# 定义参数
locations = ["Station_A", "Station_B"]  # 可以根据需求扩展多个地点
start_date = datetime(2024, 9, 1)
end_date = datetime(2024, 10, 31)
save_dir = "data"  # 保存目录

# 生成数据
generate_load_data(start_date, end_date, locations, save_dir)