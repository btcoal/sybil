import pandas as pd
import argparse
import os
import sys

def create_lagged_series(input_csv_path, date_column, series_column, max_lag, output_csv_path):
    # Attempt to read the CSV file
    df = pd.read_csv(input_csv_path, parse_dates=[date_column])
    
    # Sort the DataFrame by the date column to ensure lags are in order
    df.sort_values(by=date_column, inplace=True)
    
    # Create lagged series columns
    for lag in range(1, max_lag + 1):
        df[f'{series_column}_lag_{lag}'] = df[series_column].shift(lag)
    
    # Drop rows with NaN values introduced by shifting
    df.dropna(inplace=True)
    
    # Save the resulting DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Lagged data saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create lagged series from a CSV file.")
    parser.add_argument("-i", "--input_csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("-d", "--date_column", required=True, help="Name of the date column.")
    parser.add_argument("-s", "--series_column", required=True, help="Name of the series column.")
    parser.add_argument("-l", "--max_lag", type=int, required=True, help="Maximum number of lags to create.")
    parser.add_argument("-o", "--output_csv", required=True, help="Path to save the output CSV file.")

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: The file '{args.input_csv}' does not exist.")
        exit(1)

    create_lagged_series(args.input_csv, args.date_column, args.series_column, args.max_lag, args.output_csv)
