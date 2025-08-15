from neuralforecast.losses.pytorch import MAE, MAPE, MASE, MSE
from utils import load_and_process_data
import argparse
import pandas as pd
import torch
def main(args):
    parser = argparse.ArgumentParser(description="Train and evaluate a NeuralForecast model.")
    parser.add_argument('--forecast_path', type=str, required=True, help='Path to the input data file (.xlsx)')
    forecast_path = args.forecast_path
    forecast_df = pd.read_csv(forecast_path)
    if 