import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from neuralforecast import NeuralForecast
# We need to import the model classes to be able to load from a checkpoint
from neuralforecast.models import TimesNet, NHITS, PatchTST

from Data_processing import get_traffic_in_df, get_traffic_out_df

def get_model_class(model_name: str):
    """Returns the model class from its name."""
    model_map = {
        'TimesNet': TimesNet,
        'NHITS': NHITS,
        'PatchTST': PatchTST
    }
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_class

def main():
    parser = argparse.ArgumentParser(description="Perform rolling forecast inference with a trained model.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--model_name', type=str, required=True, choices=['TimesNet', 'NHITS', 'PatchTST'], help='Name of the model architecture.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the full input data file (.xlsx).')
    parser.add_argument('--traffic_direction', type=str, default='in', choices=['in', 'out'], help='Traffic direction to model.')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Directory to save forecasts and plots.')
    parser.add_argument('--freq', type=str, default='5min', help='Frequency of the time series data.')

    args = parser.parse_args()

    # 1. Load Model from Checkpoint
    print(f"Loading model '{args.model_name}' from: {args.checkpoint_path}")
    model_class = get_model_class(args.model_name)
    model = model_class.load_from_checkpoint(args.checkpoint_path)
    h = model.h  # Get forecast horizon from the loaded model

    # 2. Load and Prepare Data
    print(f"Loading and processing data from: {args.data_path}")
    df_raw = pd.read_excel(args.data_path, header=3)
    if args.traffic_direction == 'in':
        full_df = get_traffic_in_df(df_raw)
    else:
        full_df = get_traffic_out_df(df_raw)
    full_df.dropna(inplace=True)

    # Split data into initial history and the part to be forecasted
    train_len = int(len(full_df) * 0.8)
    history_df = full_df.iloc[:train_len].copy()
    future_df = full_df.iloc[train_len:].copy()
    print(f"Initial history size: {len(history_df)}, Future data to predict: {len(future_df)}")

    # 3. Perform Rolling Forecast
    all_forecasts = []
    print(f"Performing rolling forecast with horizon h={h}...")

    # Use tqdm for a progress bar
    for i in tqdm(range(0, len(future_df), h), desc="Rolling Forecast Steps"):
        # Create a new NeuralForecast instance for each step to ensure a clean state
        nf = NeuralForecast(models=[model], freq=args.freq)

        # Prime the forecaster with the current history.
        # max_steps=0 ensures we don't retrain; this just loads data into the model's context.
        nf.fit(df=history_df, val_size=0, max_steps=0)

        # Predict h steps into the future from the end of the current history
        forecast = nf.predict()
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
    print(f"Forecasts saved to: {forecast_csv_path}")

    # 5. Plot and Save Figure
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_df = results_df.set_index('ds')

    plot_df['y'].plot(ax=ax, label='Actual')
    plot_df[args.model_name].plot(ax=ax, label='Rolling Forecast', linestyle='--')

    ax.set_title(f'Rolling Forecast vs Actuals ({args.model_name})')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Traffic (Mbps)')
    ax.legend()
    ax.grid(True)

    plot_path = os.path.join(run_output_dir, 'rolling_forecast_plot.png')
    fig.savefig(plot_path)
    print(f"Forecast plot saved to: {plot_path}")
    plt.close(fig)

if __name__ == '__main__':
    main()
