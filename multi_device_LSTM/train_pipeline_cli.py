import argparse
import sys
import importlib.util
import os
import pandas as pd

def import_module(file_path):
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description="Split dataset and train the LSTM model for time series prediction.")
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Directory containing original datasets')
    parser.add_argument('--model_save_dir', type=str, default='.', help='Directory to save the best models')
    
    args = parser.parse_args()

    # Import the dataset_split.py module
    dataset_split_module = import_module("dataset_split.py")

    # Import the train module
    train_module = import_module("train.py")

    # Get all CSV files in the input directory
    input_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('_origin_dataset.csv')]

    devices = []

    for input_file in input_files:
        device_name = input_file.replace('_origin_dataset.csv', '')
        devices.append(device_name)
        print(f"Processing device: {device_name}")

        # Read the original dataset
        df = pd.read_csv(os.path.join(args.dataset_dir, input_file))

        # Run dataset split
        dataset_split_module.split_dataset(device_name, df)

        print(f"Dataset split completed for {device_name}.")

    print("Dataset split completed for all devices.")

    # Run model training for all devices
    for device_name in devices:
        print(f"Training model for device: {device_name}")
        train_module.train_model(device_name)

    print("Training completed for all devices.")

if __name__ == "__main__":
    main()