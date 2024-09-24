## 数据集要求

### 单设备情况

```bash
dataset/
	origin_dataset.csv
```

```bash
timestamp,load_value
2020-01-01 00:00:00,244104.7682
2020-01-01 00:15:00,232760.6402
2020-01-01 00:30:00,232041.6891
2020-01-01 00:45:00,227353.3317
2020-01-01 01:00:00,232994.4269
2020-01-01 01:15:00,224198.7554
```

需要按15分钟的间隔记录每个地点的负荷数据，从00:00:00到23:45:00。

**请确保数据的格式正确！使用原始值即可，代码内会进行归一化处理。**

### 多设备情况

```bash
dataset/
    device1_origin_dataset.csv
    device2_origin_dataset.csv
    ...
```

其余部分与单设备情况基本一致。

代码会按顺序把dataset里的所有device进行训练，分别导出模型。

**csv文件命名时，原始数据请命名为"device_origin_dataset.csv"，以"_origin_dataset.csv"结尾！！！**

**附：调试代码过程所使用的模拟数据来源：https://tianchi.aliyun.com/dataset/176964**

## 接口文件说明(以多设备情况为例)

**共形成两个接口文件，train_pipeline_cli.py 和 load_prediction_cli.py，其余文件用于单独对代码、模型进行测试，可选用。**

### 一、模型训练代码

#### 基本命令格式：
```bash
python train_pipeline.py [--dataset_dir DATASET_DIR] [--model_save_dir MODEL_SAVE_DIR]
```

#### 参数说明：
- `--dataset_dir`：可选参数，指定存放原始数据集的目录。默认值是 `dataset`。该目录中的每个文件名必须以 `_origin_dataset.csv` 结尾，且文件名的前缀部分应代表设备名。
- `--model_save_dir`：可选参数，指定保存模型的目录。默认值是当前目录 `.`。

#### 示例命令：
1. **使用默认目录处理所有设备的数据并训练模型**：
   ```bash
   python train_pipeline_cli.py
   ```
   - 该命令会从默认的 `dataset` 目录中获取所有符合条件的设备数据，拆分数据集并开始训练。

2. **指定数据集目录和模型保存目录**：
   ```bash
   python train_pipeline_cli.py --dataset_dir my_datasets --model_save_dir my_models
   ```
   - 该命令会从 `my_datasets` 目录读取设备数据，并将训练后的模型保存在 `my_models` 目录中。

3. **仅指定数据集目录**：
   ```bash
   python train_pipeline_cli.py --dataset_dir custom_dataset_folder
   ```
   - 该命令会从 `custom_dataset_folder` 目录中读取设备数据，并在当前目录中保存训练的模型。

### 注意事项
1. 数据文件必须命名为 `{device_name}_origin_dataset.csv` 格式，其中 `{device_name}` 是设备的名称，`_origin_dataset.csv` 是文件的后缀。
2. 脚本将会自动调用 `dataset_split.py` 中的 `split_dataset` 函数，以及 `train.py` 中的 `train_model` 函数，请确保这两个模块存在并且其函数正确实现。
3. 如果有多个设备数据文件，脚本会循环处理每个设备，依次拆分数据并训练模型。

### 示例工作流程：
- 设备 `device1` 的数据文件名为 `device1_origin_dataset.csv`。
- 设备 `device2` 的数据文件名为 `device2_origin_dataset.csv`。
  

执行以下命令：
```bash
python train_pipeline_cli.py --dataset_dir dataset --model_save_dir models
```
将会拆分每个设备的数据，并为每个设备分别训练模型，并保存在 `models` 目录中。

### 二、模型预测代码

1. **基本命令格式**：
   ```bash
   python load_prediction_cli.py "YYYY-MM-DD HH:MM:SS" [--num_days N] [--devices device1 device2 ...] [--output_dir directory]
   ```

2. **参数说明**：
   - `start_date`：预测开始日期，格式为 `"YYYY-MM-DD HH:MM:SS"`，例如 `"2023-01-16 00:00:00"`。
   - `--num_days`：可选参数，指定预测的天数，取值为 `1`、`3`、`30` 或 `90`。如果不指定，将会为所有可选值生成预测文件。
   - `--devices`：可选参数，指定要预测的设备名称列表，例如 `device1 device2`。如果不提供，将自动从数据集中获取所有设备。
   - `--output_dir`：可选参数，指定输出目录，默认是 `predictions`。

3. **示例命令**：
   - **预测未来3天的负载**：
     
     ```bash
     python load_prediction_cli.py "2023-01-16 00:00:00" --num_days 3 --devices device1 device2
     ```
   - **为所有设备预测未来30天的负载**（自动从数据集中获取设备）：
     ```bash
     python load_prediction_cli.py "2023-01-16 00:00:00" --num_days 30
     ```
   - **为所有设备预测负载，输出到自定义目录**：
     ```bash
     python load_prediction_cli.py "2023-01-16 00:00:00" --output_dir my_predictions
     ```
   - **为所有设备生成所有天数的预测文件**（未指定 `num_days`）：
     ```bash
     python load_prediction_cli.py "2023-01-16 00:00:00" --devices device1 device2
     ```

## Reference

**感谢Garyou19开源的代码，本项目在此基础上进一步开发。**

**原仓库链接：https://github.com/Garyou19/LSTM_PyTorch_Electric-Load-Forecasting**
