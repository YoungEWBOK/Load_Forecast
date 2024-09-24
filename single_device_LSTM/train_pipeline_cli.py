import argparse
import sys
import importlib.util
import os

def import_module(file_path):
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description="Split dataset and train the LSTM model for time series prediction.")
    parser.add_argument('--input_data', type=str, default='dataset/origin_dataset.csv', help='Path to the original dataset')
    parser.add_argument('--output_dir', type=str, default='dataset', help='Directory to save split datasets')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of the training dataset to include in the validation split')
    parser.add_argument('--model_save_path', type=str, default='best_model.pt', help='Path to save the best model')
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length for input data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of the LSTM model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

    args = parser.parse_args()

    # Import the dataset_split.py module
    dataset_split_module = import_module("dataset_split.py")

    # Run dataset split
    dataset_split_module.split_dataset(args.input_data, args.output_dir, args.test_size, args.val_size)

    print("Dataset split completed.")

    # 导入 train 模块
    train_module = import_module("train.py")

    # 设置训练集和验证集的路径
    train_data_path = os.path.join(args.output_dir, 'train_dataset.csv')
    val_data_path = os.path.join(args.output_dir, 'val_dataset.csv')

    # 运行模型训练
    train_module.train_model(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        model_save_path=args.model_save_path,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

    print(f"Training completed. Best model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()