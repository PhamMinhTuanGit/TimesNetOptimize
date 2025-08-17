import argparse
import os
import pandas as pd
from tqdm import tqdm
import logging

from neuralforecast import NeuralForecast
# We need to import the model classes to be able to load from a checkpoint
from utils import load_and_process_data


def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up and returns the argument parser."""
    parser = argparse.ArgumentParser(description="Perform rolling forecast inference with a trained model.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model, used for naming output files.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the full input data file (.xlsx).')
    parser.add_argument('--traffic_direction', type=str, default='in', choices=['in', 'out'], help='Traffic direction to model.')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Directory to save forecasts and plots.')
    parser.add_argument('--silent', action='store_true', help='If set, disables all logging and progress bars.')
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
    run_output_dir = os.path.join(output_dir, f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Save CSV
    forecast_csv_path = os.path.join(run_output_dir, 'rolling_forecast.csv')
    results_df.to_csv(forecast_csv_path, index=False)
    return forecast_csv_path


def main():
    """Main function to run the inference pipeline."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.silent:
        logging.getLogger("neuralforecast").setLevel(logging.ERROR)
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    if not args.silent:
        print(f"Loading model '{args.model_name}' from: {args.checkpoint_path}")
    nf = NeuralForecast.load(path=args.checkpoint_path)

    history_df, future_df = load_and_process_data(args.data_path, args.traffic_direction, train_scale=0.1)
    if not args.silent:
        print(f"Initial history size: {len(history_df)}, Future data to predict: {len(future_df)}")

    forecasts_df = perform_rolling_forecast(nf, history_df, future_df, args.silent)

    forecast_csv_path = save_results(forecasts_df, future_df, args.output_dir, args.model_name, args.silent)

    # The final print is the path to the forecast CSV, for scripting.
    print(forecast_csv_path)


if __name__ == '__main__':
    main()
