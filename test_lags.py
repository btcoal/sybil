import pandas as pd
import pytest
import os
import sys
from lags import create_lagged_series

# Define test paths for creating temporary files
TEST_CSV_PATH = "test_input.csv"
OUTPUT_CSV_PATH = "test_output.csv"

# Sample data to use in the test CSV
TEST_DATA = {
    "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    "Close": [100, 101, 102, 103, 104]
}
EXPECTED_COLUMNS = ["Date", "Close", "Close_lag_1", "Close_lag_2"]

@pytest.fixture(scope="function")
def setup_csv():
    """Create a temporary CSV file for testing and cleanup afterward."""
    df = pd.DataFrame(TEST_DATA)
    df.to_csv(TEST_CSV_PATH, index=False)
    yield TEST_CSV_PATH  # Yield control for the test function to use
    # Cleanup after tests
    if os.path.exists(TEST_CSV_PATH):
        os.remove(TEST_CSV_PATH)
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)

def test_create_lagged_series(setup_csv):
    """Test the create_lagged_series function with a valid input file."""
    # Call the function with a valid test CSV file
    create_lagged_series(setup_csv, "Date", "Close", 2, OUTPUT_CSV_PATH)

    # Check if the output file is created
    assert os.path.exists(OUTPUT_CSV_PATH), "Output CSV file was not created."

    # Load the output file to verify its contents
    df_output = pd.read_csv(OUTPUT_CSV_PATH)

    # Check that the lagged columns are created and have the correct names
    for column in EXPECTED_COLUMNS:
        assert column in df_output.columns, f"Column '{column}' is missing in the output CSV."

    # Verify that the correct number of rows is present (with dropped NaNs)
    expected_row_count = len(TEST_DATA["Date"]) - 2  # Original length minus max_lag
    assert len(df_output) == expected_row_count, "Output CSV has an incorrect number of rows."

    # Check if the lag values are correct
    assert df_output["Close_lag_1"].iloc[0] == 101, "Incorrect value in Close_lag_1."
    assert df_output["Close_lag_2"].iloc[0] == 100, "Incorrect value in Close_lag_2."

def test_invalid_input_file():
    """Test function with an invalid input file path."""
    with pytest.raises(FileNotFoundError):
        create_lagged_series("non_existent.csv", "Date", "Close", 2, OUTPUT_CSV_PATH)
