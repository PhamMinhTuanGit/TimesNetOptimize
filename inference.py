import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from neuralforecast import NeuralForecast
# We need to import the model classes to be able to load from a checkpoint

from model import add_model_args, create_model_from_args, count_parameters
from utils import save_dict_to_json, load_and_process_data


def main():
    parser = argparse.ArgumentParser(description="Perform rolling forecast inference with a trained model.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the full input data file (.xlsx).')
    parser.add_argument('--traffic_direction', type=str, default='in', choices=['in', 'out'], help='Traffic direction to model.')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Directory to save forecasts and plots.')
    parser.add_argument('--freq', type=str, default='5min', help='Frequency of the time series data.')
    parser = add_model_args(parser)
    args = parser.parse_args()
    
    # 1. Load Model from Checkpoint
    print(f"Loading model '{args.model_name}' from: {args.checkpoint_path}")
    model = create_model_from_args(args=args)
    h = model.h  # Get forecast horizon from the loaded model
    nf = NeuralForecast(models=[model], freq=args.freq)
    nf = NeuralForecast.load(path = args.checkpoint_path)
    # 2. Load and Prepare Data
    history_df, future_df = load_and_process_data(args.data_path, args.traffic_direction, train_scale = 0.1)
    # Split data into initial history and the part to be forecasted
    
    print(f"Initial history size: {len(history_df)}, Future data to predict: {len(future_df)}")

    # 3. Perform Rolling Forecast
    all_forecasts = []
    print(f"Performing rolling forecast with horizon h={h}...")
    
    # Use tqdm for a progress bar
    for i in tqdm(range(0, len(future_df), h), desc="Rolling Forecast Steps"):
        # Create a new NeuralForecast instance for each step to ensure a clean state
        

        # Prime the forecaster with the current history.
        # max_steps=0 ensures we don't retrain; this just loads data into the model's context

        # Predict h steps into the future from the end of the current history
        
        forecast = nf.predict(df = history_df)
        all_forecasts.append(forecast)

        # Update history by appending the *actual* observed data for the next iteration
        actuals_for_step = future_df.iloc[i : i + h]
        if actuals_for_step.empty:
            break
        history_df = pd.concat([history_df, actuals_for_step])

    # 4. Process and Save Results
    print("Processing and saving results...")
    # Combine all forecast chunks
    forecasts_df = pd.concat(all_forecasts).reset_index()

    # Merge with actuals from the future_df for comparison
    results_df = pd.merge(future_df, forecasts_df, on=['unique_id', 'ds'], how='left')
    results_df.dropna(inplace=True)  # Drop any rows where prediction wasn't possible

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    run_output_dir = os.path.join(args.output_dir, f"{args.model_name}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Save CSV
    forecast_csv_path = os.path.join(run_output_dir, 'rolling_forecast.csv')
    results_df.to_csv(forecast_csv_path, index=False)
    # The final print is the path to the forecast CSV, for scripting.
    print(forecast_csv_path)


if __name__ == '__main__':
    main()
