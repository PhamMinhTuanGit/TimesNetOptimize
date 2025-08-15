import pandas as pd
import torch
import os
import argparse
import matplotlib.pyplot as plt
import time

from neuralforecast import NeuralForecast
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from utils import load_and_process_data

# Import from your local modules

from model import add_model_args, create_model_from_args, count_parameters
from utils import save_dict_to_json

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Train and evaluate a NeuralForecast model.")
    
    # --- Add Training-specific arguments ---
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file (.xlsx)')
    parser.add_argument('--traffic_direction', type=str, default='in', choices=['in', 'out'], help='Traffic direction to model.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save logs, checkpoints, and forecasts.')
    parser.add_argument('--freq', type=str, default='5min', help='Frequency of the time series data.')
    parser.add_argument('--val_size', type=int, default=12, help='Size of the validation set for NeuralForecast.')
    parser.add_argument('--disable_checkpointing', action='store_true', help='If set, disables saving model checkpoints.')

    # --- Add Model-specific arguments from model.py ---
    parser = add_model_args(parser)

    # --- Parse all arguments ---
    args = parser.parse_args()

    # 2. Load and Prepare Data
    train_df, test_df = load_and_process_data(args.data_path, args.traffic_direction, train_scale=1)

    # 3. Setup Logging and Checkpoints
    run_output_dir = os.path.join(args.output_dir, args.model_name, f"run_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_output_dir, exist_ok=True)

  

    # 4. Create the Model using the unified arguments, passing logger and callbacks
    print(f"\n--- Creating Model: {args.model_name} ---")
    model = create_model_from_args(
        args=args,
        callbacks=None,
        logger=None
    )
    nf = NeuralForecast(models=[model], freq=args.freq)
    print("\n--- Starting Training ---")
    params = count_parameters(model)/1e6
    nf.fit(df=train_df, val_size=args.val_size)

    ckpt_path = f'output/{args.model_name}/{args.model_name}_{params:.1f}m'
    nf.save(path = ckpt_path, overwrite=True)
    print(ckpt_path)

    

    




if __name__ == '__main__':
    main()
