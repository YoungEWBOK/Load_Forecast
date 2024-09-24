# 负荷预测代码说明

## 文件目录

```bash
load_predict/ 
├── data/
│   ├── 2024-09/
│   │   ├── day1.csv
│   │   ├── day2.csv
│   │   └── ...
│   └── 2024-10/
│       ├── day1.csv
│       ├── day2.csv
│       └── ...
├── predictions/
│   ├── predictions_combined_all_locations_20241101_to_20241107_created_20240922.csv
│   ├── predictions_Station_A_20241101_to_20241107_created_20240922.csv
│   └── predictions_Station_B_20241101_to_20241107_created_20240922.csv
├── generate_data.py
├── lstm_model_Station_A.pth
├── lstm_model_Station_B.pth
├── predict.py
├── run.sh
├── scaler_Station_A.pth
├── scaler_Station_B.pth
└── train_model.py
```

## 各文件使用方式

### data文件夹

data文件夹里按照月份和日期的顺序组织所有数据，CSV格式大致如下：

| timestamp           | location  | load_value |
| ------------------- | --------- | ---------- |
| 2024-09-01 00:00:00 | Station_A | 105.32     |
| 2024-09-01 00:15:00 | Station_A | 128.12     |
| 2024-09-01 00:30:00 | Station_A | 146.01     |

按15分钟的间隔记录每个地点的负荷数据，从00:00:00到23:45:00。

### predictions文件夹

此文件夹存储预测负荷值的输出，包括每个地点的单独输出和一份合并文件。

文件名格式为：`predictions_location_start_to_end_created_current.csv`

包含的信息有：

- 位置名称（对于合并文件，使用 "all_locations"）
- 预测的起始日期
- 预测的结束日期
- 文件创建的日期

### generate_data.py

用于生成模拟数据，主要在导入真实数据时使用，真实数据导入后可不再使用该脚本。

### train_model.py

训练脚本中设置了 **时间步长度**（即在预测某个时间点的负荷时，模型考虑前 `seq_length` 个时间步的数据）和 **训练轮次** `epochs`。默认情况下，会读取所有历史数据以捕捉时间序列中的长期趋势。如果不想使用全部数据，可以按需放入data文件夹。

该脚本会根据位置（location）分类，单独训练模型，生成每个地点的 **模型权重文件（.pth）** 和 **数据标准化对象的 scaler.pth 文件**。这些文件在 `predict.py` 中用于数据标准化，例如，`lstm_model_Station_A.pth` 和 `scaler_Station_A.pth` 是训练后生成的输出文件。

### predict.py

预测脚本中设置了 `seq_length` 和 `future_days`（预测未来几天的负荷值，默认值为7，按需调整）。该脚本会使用训练好的模型和对应的 scaler 对新数据进行负荷预测。