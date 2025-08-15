import json
import pandas as pd
import numpy as np
from Data_processing import process_traffic_data
from neuralforecast.losses.pytorch import MAE, MAPE, SMAPE, MASE, MSE
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE, avoiding division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by replacing 0s in y_true with a small number or 1
    # Here we replace with 1, assuming traffic values are significantly larger.
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

def save_dict_to_json(data: dict, file_path: str):
    """Saves a dictionary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_and_process_data(file_path: str, traffic_direction: str = 'in', train_scale = 0.8):
    """Loads and processes data from the specified Excel file."""
    print(f"Loading data from: {file_path}")
    df_raw = pd.read_excel(file_path, header=3)
    df_raw.columns = df_raw.columns.str.strip()

    print(f"Processing traffic direction: {traffic_direction}")
    df = process_traffic_data(df_raw, direction=traffic_direction)

    df.dropna(inplace=True)

    # Split data
    total_len = len(df)
    train_len = int(total_len * train_scale)
    train_df = df.iloc[:train_len]
    test_df = df.iloc[train_len:]

    print(f"Data loaded successfully. Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df