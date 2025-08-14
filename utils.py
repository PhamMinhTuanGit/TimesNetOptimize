import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE, avoiding division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by replacing 0s in y_true with a small number or 1
    # Here we replace with 1, assuming traffic values are significantly larger.
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

def calculate_metrics(df: pd.DataFrame, model_name: str) -> dict:
    """Calculates and returns a dictionary of evaluation metrics."""
    df = df.dropna(subset=['y', model_name])
    if df.empty:
        return {}
        
    y_true = df['y']
    y_pred = df[model_name]
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }
    return {k: float(v) for k, v in metrics.items()} # Ensure JSON serializable

def save_dict_to_json(data: dict, file_path: str):
    """Saves a dictionary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)