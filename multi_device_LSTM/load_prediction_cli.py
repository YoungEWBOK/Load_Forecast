import argparse
from custom_finetune import predict_future
import pandas as pd
import os

def predict_and_export_to_csv(device_name, start_date, num_days, output_file, model_path):
    results = predict_future(device_name, start_date, num_days, model_path=model_path)
    
    results.to_csv(output_file, index=False)
    print(f"Results for {device_name} exported to {output_file}")
    
    # Print preview of results
    print(f"Preview of results for {device_name}:")
    print(results.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict load and export to CSV for multiple devices.")
    parser.add_argument("start_date", type=str, help="Start date in format 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--num_days", type=int, choices=[1, 3, 30, 90], help="Number of days to predict")
    parser.add_argument("--devices", type=str, nargs='+', help="List of device names to predict for")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Output directory for CSV files")
    
    args = parser.parse_args()

    # If no devices are specified, use all devices found in the dataset directory
    if not args.devices:
        dataset_dir = 'dataset'
        args.devices = [f.replace('_test_dataset.csv', '') for f in os.listdir(dataset_dir) if f.endswith('_test_dataset.csv')]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # If num_days is not provided, loop through all choices [1, 3, 30, 90]
    choices = [1, 3, 30, 90]
    if args.num_days is None:
        for num_days in choices:
            for device_name in args.devices:
                output_file = os.path.join(args.output_dir, f'{device_name}_predictions_{num_days}days.csv')
                
                # Use {device_name}_best_model.pt as the model path
                model_path = f'{device_name}_best_model.pt'
                
                predict_and_export_to_csv(device_name, args.start_date, num_days, output_file, model_path)
    else:
        # If num_days is provided, run the prediction only for that number of days
        for device_name in args.devices:
            output_file = os.path.join(args.output_dir, f'{device_name}_predictions_{args.num_days}days.csv')
            
            # Use {device_name}_best_model.pt as the model path
            model_path = f'{device_name}_best_model.pt'
            
            predict_and_export_to_csv(device_name, args.start_date, args.num_days, output_file, model_path)

    print("All predictions completed.")