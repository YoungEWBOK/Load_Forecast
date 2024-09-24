## 单设备情况

### 数据要求

```
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

模拟数据来源：https://tianchi.aliyun.com/dataset/176964

### 代码说明

### 1. **文件结构**

项目主要包含三个核心文件：

- `dataset_split.py`：用于数据预处理和划分训练、测试集。
- `train.py`：用于模型的训练，包括模型定义和训练循环。
- `test.py`：用于加载训练好的模型并进行预测。

### 2. **运行环境设置**

确保你在合适的环境中运行代码，推荐使用 `conda` 管理环境。在项目开始前，请按照以下步骤设置环境：

1. 创建并激活 `conda` 环境：

   ```bash
   conda create -n load_forecasting python=3.8
   conda activate load_forecasting
   ```

2. 安装必要的依赖：

   ```bash
   pip install torch scikit-learn numpy pandas
   ```

### 3. **数据准备与划分 (`dataset_split.py`)**

在开始训练模型之前，首先需要对原始数据进行处理和划分。该文件负责将数据集按照训练集和测试集的比例进行拆分。请确保数据包含必要的列，例如 `timestamp` 和 `load_value`。

#### 使用方法：

运行以下命令来进行数据划分：

```bash
python dataset_split.py
```

该脚本将自动读取数据文件，处理后生成训练集和测试集文件。

**划分后dataset文件夹如下所示**

```bash
dataset/
	origin_dataset.csv
    train_dataset.csv
    val_dataset.csv
    test_dataset.csv
```

### 4. **训练模型 (`train.py`)**

在数据集划分后，使用 `train.py` 文件进行模型训练。文件中定义了LSTM等模型结构，并实现了训练循环。在训练过程中，模型会不断评估验证集上的表现，保存性能最好的模型。

#### 使用方法：

```bash
python train.py
```

训练完成后，最佳模型将保存为 `best_model.pt`，并可以在预测阶段加载使用。

### 5. **预测 (`test.py`)**

`test.py` 用于加载训练好的模型并对测试集或新数据进行负荷预测。在该脚本中，模型会从 `best_model.pt` 文件加载，如果是在CPU上运行，需要注意使用 `map_location=torch.device('cpu')`。

#### 使用方法：

```bash
python test.py
```

该脚本会输出预测结果，并与真实的负荷值进行比较，帮助评估模型的准确性。

## 多设备情况

请确保数据集划分后，dataset的目录结构如下：

```bash
dataset/
    device1_train_dataset.csv
    device1_val_dataset.csv
    device1_test_dataset.csv
    device2_train_dataset.csv
    device2_val_dataset.csv
    device2_test_dataset.csv
    ...
```

其余部分与单设备情况基本一致。

代码会按顺序把dataset里的所有device进行训练，分别导出模型。

**csv文件命名时，原始数据请命名为"device_origin_dataset.csv"，以"_origin_dataset.csv"结尾！！！**