import argparse
import os
import pandas as pd
from tqdm import tqdm
import logging
import warnings
import lightning_utilities.core.rank_zero as rank_zero
def silent(*args, **kwargs): pass
rank_zero.rank_zero_info = silent
rank_zero.rank_zero_warn = silent
from neuralforecast import NeuralForecast
# We need to import the model classes to be able to load from a checkpoint
from utils import load_and_process_data
os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"
os.environ["PL_DISABLE_LOGGING"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up and returns the argument parser."""
    parser = argparse.ArgumentParser(description="Perform rolling forecast inference with a trained model.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model, used for naming output files.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the full input data file (.xlsx).')
    parser.add_argument('--traffic_direction', type=str, default='in', choices=['in', 'out'], help='Traffic direction to model.')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Directory to save forecasts and plots.')
    parser.add_argument('--silent', action='store_true', help='If set, disables all logging and progress bars.')
    parser.add_argument('--training_path', required=True, help = 'History dataFrame to fit the model')
    return parser


def perform_rolling_forecast(nf: NeuralForecast, history_df: pd.DataFrame, future_df: pd.DataFrame, silent: bool) -> pd.DataFrame:
    """
    Performs a rolling forecast simulation.

    Args:
        nf: The initialized NeuralForecast object.
        history_df: The initial DataFrame to start forecasting from.
        future_df: The DataFrame with future actual values to iterate over.
        silent: Flag to disable progress bar.

    Returns:
        A DataFrame containing all concatenated forecasts.
    """
    h = nf.models[0].h  # Get forecast horizon from the loaded model
    all_forecasts = []

    if not silent:
        print(f"Performing rolling forecast with horizon h={h}...")

    # Use tqdm for a progress bar, which is disabled in silent mode
    for i in tqdm(range(0, len(future_df), h), desc="Rolling Forecast Steps", disable=silent):
        # Predict h steps into the future from the end of the current history
        forecast = nf.predict(df=history_df)
        all_forecasts.append(forecast)

        # Update history by appending the *actual* observed data for the next iteration
        actuals_for_step = future_df.iloc[i: i + h]
        if actuals_for_step.empty:
            break
        history_df = pd.concat([history_df, actuals_for_step])
        print(f'Progress: {i}/{len(future_df)}')
    return pd.concat(all_forecasts).reset_index()


def save_results(forecasts_df: pd.DataFrame, future_df: pd.DataFrame, output_dir: str, model_name: str, silent: bool) -> str:
    """
    Merges forecasts with actuals, saves to a CSV, and returns the file path.
    """
    if not silent:
        print("Processing and saving results...")

    # Merge with actuals from the future_df for comparison
    results_df = pd.merge(future_df, forecasts_df, on=['unique_id', 'ds'], how='left')
    results_df.dropna(inplace=True)  # Drop any rows where prediction wasn't possible

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    

    # Save CSV
    forecast_csv_path = os.path.join(output_dir, f'rolling_forecast_{model_name}.csv')
    results_df.to_csv(forecast_csv_path, index=False)
    return forecast_csv_path




def main():
    """Main function to run the inference pipeline."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Suppress logging from libraries
    logging.getLogger("neuralforecast").setLevel(logging.CRITICAL)
    logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
    logging.getLogger("lightning").setLevel(logging.CRITICAL)

    # Environment variables to silence PyTorch Lightning CLI and other outputs
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["PL_DISABLE_LOGGING"] = "1"

    if not args.silent:
        print(f"Loading model '{args.model_name}' from: {args.checkpoint_path}")
    nf = NeuralForecast.load(path=args.checkpoint_path)

    future_df, _ = load_and_process_data(args.data_path, args.traffic_direction, train_scale=1)
    history_df, _ = load_and_process_data(args.training_path, args.traffic_direction, train_scale=1)
    if not args.silent:
        print(f"Initial history size: {len(history_df)}, Future data to predict: {len(future_df)}")

    forecasts_df = perform_rolling_forecast(nf, history_df, future_df, args.silent)

    forecast_csv_path = save_results(forecasts_df, future_df, args.output_dir, args.model_name, args.silent)

    # Final output: path to forecast CSV
    print(forecast_csv_path)

if __name__ == '__main__':
    main()
