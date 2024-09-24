'''
最少需要指定开始时间这个参数，后面也可以调整成已有数据最后一个时间戳的下一时间点
python load_prediction_cli.py "2023-01-16 00:00:00"
'''
import argparse
from custom_finetune import predict_future
import pandas as pd
from datetime import datetime, timedelta
import os

def predict_and_export_to_csv(start_date, num_days, output_file):
    results = predict_future(start_date, num_days)
    
    results.to_csv(output_file, index=False)
    print(f"Results exported to {output_file}")
    
    # Print preview of results
    print("Preview of results:")
    print(results.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict load and export to CSV.")
    parser.add_argument("start_date", type=str, help="Start date in format 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--num_days", type=int, choices=[1, 3, 30, 90], help="Number of days to predict")
    parser.add_argument("--output", type=str, help="Output CSV file name", default=None)

    args = parser.parse_args()

    # Ensure prediction folder exists
    output_folder = "predictions"
    os.makedirs(output_folder, exist_ok=True)

    if args.num_days is None:
        choices = [1, 3, 30, 90]
        for num_days in choices:
            output_file = os.path.join(output_folder, f'predictions_{num_days}days.csv')
            predict_and_export_to_csv(args.start_date, num_days, output_file)
    else:
        if args.output is None:
            args.output = os.path.join(output_folder, f'predictions_{args.num_days}days.csv')
        
        predict_and_export_to_csv(args.start_date, args.num_days, args.output)