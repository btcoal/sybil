"""
Take a time series and return the lags of the time series.

Arguments:
- fp: str, file path to the time series
- lags: int, number of lags to return
- file: str, optional, file path to save the lags

Returns:
File with the lags of the time series.
"""

import numpy as np
import pandas as pd
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Return the lags of a time series')
parser.add_argument('--fp', type=str, required=True, help='file path to the time series')
parser.add_argument('--header', type=int, default=0, help='header row (0 for header, None for no header)')
parser.add_argument('--var', type=str, required=True, help='variable name of the time series')
parser.add_argument('--lags', type=int, required=True, help='number of lags to return')
parser.add_argument('--file', type=str, help='file path to save the lags')
args = parser.parse_args()

# Create save filepath
if args.file:
    save_path = args.file + '.csv'
else:
    save_path = args.fp.split('.')[0] + '_lags.csv'

# Print the arguments
print('File path:', args.fp)
print('Header:', args.header)
print('Variable:', args.var)
print('Lags:', args.lags)
print('Save file:', save_path)

# Load the time series
ts = pd.read_csv(args.fp, header=args.header)
if args.var not in ts.columns:
    raise ValueError(f"Variable '{args.var}' not found in the file.")

ts_values = ts[args.var].values

# Create the lags
n = len(ts_values)
lags = np.full((n, args.lags), np.nan)
for i in range(args.lags):
    lags[i:, i] = ts_values[:n - i]

# Convert to DataFrame and save
lagged_df = pd.DataFrame(lags, columns=[f'{args.var}_lag_{i+1}' for i in range(args.lags)])
lagged_df.to_csv(save_path, index=False)
print(f"Lagged time series saved to {save_path}")
