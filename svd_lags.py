import pandas as pd
import numpy as np
import argparse
import os

def create_lagged_matrix(df, date_column, series_column, max_lag):
    """Creates a matrix of lagged values for a given series in the DataFrame."""
    df.sort_values(by=date_column, inplace=True)
    
    # Create lagged columns
    for lag in range(1, max_lag + 1):
        df[f'{series_column}_lag_{lag}'] = df[series_column].shift(lag)
    
    # Drop rows with NaN values introduced by shifting
    df.dropna(inplace=True)
    
    # Extract only the lagged columns for the SVD
    lagged_columns = [f'{series_column}_lag_{lag}' for lag in range(1, max_lag + 1)]
    return df[lagged_columns]

def perform_svd(matrix):
    """Performs SVD on the input matrix using numpy and returns the decomposed components."""
    U, Sigma, VT = np.linalg.svd(matrix, full_matrices=False)
    return U, Sigma, VT

def main(input_csv, date_column, series_column, max_lag, output_csv, n_components):
    # Read the CSV file
    df = pd.read_csv(input_csv, parse_dates=[date_column])
    
    # Create the lagged matrix
    lagged_matrix = create_lagged_matrix(df, date_column, series_column, max_lag)
    
    # Perform SVD
    U, Sigma, VT = perform_svd(lagged_matrix)
    
    # Trim U matrix to the desired number of components
    U_trimmed = U[:, :n_components]
    Sigma_trimmed = Sigma[:n_components]
    VT_trimmed = VT[:n_components, :]
    
    # Save U matrix to output CSV for inspection
    output_df = pd.DataFrame(U_trimmed, columns=[f'U_{i+1}' for i in range(U_trimmed.shape[1])])
    output_df.to_csv(output_csv, index=False)
    
    print(f"SVD complete. U matrix saved to {output_csv}")
    print("Singular values:")
    print(Sigma_trimmed)
    print("Right singular vectors (transposed):")
    print(VT_trimmed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lags from a CSV and perform SVD.")
    parser.add_argument("-i", "--input_csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("-d", "--date_column", required=True, help="Name of the date column.")
    parser.add_argument("-s", "--series_column", required=True, help="Name of the series column.")
    parser.add_argument("-l", "--max_lag", type=int, required=True, help="Maximum number of lags to create.")
    parser.add_argument("-o", "--output_csv", required=True, help="Path to save the U matrix CSV.")
    parser.add_argument("-n", "--n_components", type=int, default=2, help="Number of SVD components to keep.")

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: The file '{args.input_csv}' does not exist.")
        exit(1)

    main(args.input_csv, args.date_column, args.series_column, args.max_lag, args.output_csv, args.n_components)
