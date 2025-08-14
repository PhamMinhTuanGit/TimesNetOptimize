import pandas as pd
import torch
import os
import argparse
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Import from your local modules
from Data_processing import process_traffic_data
from model import add_model_args, create_model_from_args


def load_and_process_data(file_path: str, traffic_direction: str = 'in'):
    """Loads and processes data from the specified Excel file."""
    print(f"Loading data from: {file_path}")
    df_raw = pd.read_excel(file_path, header=0)
    df_raw.columns = df_raw.columns.str.strip()

    print(f"Processing traffic direction: {traffic_direction}")
    df = process_traffic_data(df_raw, direction=traffic_direction)

    # Drop rows with NaN values that might have been created during processing
    df.dropna(inplace=True)

    # Split data
    total_len = len(df)
    train_len = int(total_len * 0.8)
    train_df = df.iloc[:train_len]
    test_df = df.iloc[train_len:]

    print(f"Data loaded successfully. Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df


def main():
    # 1. Setup Argument Parser
    # We use parse_known_args to separate training-specific args from model-specific args
    parser = argparse.ArgumentParser(description="Train a NeuralForecast model.")

    # --- Training and Data Arguments ---
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file (e.g., .xlsx)')
    parser.add_argument('--traffic_direction', type=str, default='in', choices=['in', 'out'], help='Traffic direction to model.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save logs, checkpoints, and forecasts.')
    parser.add_argument('--freq', type=str, default='5min', help='Frequency of the time series data.')
    parser.add_argument('--val_size', type=int, default=12, help='Size of the validation set for NeuralForecast.')

    # Parse only the arguments defined above. The rest are for the model.
    train_args, model_args = parser.parse_known_args()

    # 2. Load and Prepare Data
    train_df, test_df = load_and_process_data(train_args.data_path, train_args.traffic_direction)

    # 3. Create the Model using model.py's main function
    # Find the model name from the arguments passed to the model
    # A more robust way to find the model name without fully parsing all model args here
    temp_parser = argparse.ArgumentParser()
    temp_parser.add_argument('--model_name')
    model_name_args, _ = temp_parser.parse_known_args(model_args)
    model_name = model_name_args.model_name
    if not model_name:
        raise ValueError("Error: --model_name argument is required. Eg: --model_name TimesNet")

    print(f"\n--- Creating Model: {model_name} ---")
    model = create_model(model_args)

    # 4. Setup Logging and Checkpoints
    run_output_dir = os.path.join(train_args.output_dir, model_name, f"run_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_output_dir, exist_ok=True)

    logger = TensorBoardLogger(save_dir=train_args.output_dir, name=model_name)
    checkpoints_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    # Add callbacks and logger to the model instance
    model.callbacks = [checkpoint_callback]
    model.logger = logger

    # 5. Initialize and Train NeuralForecast
    nf = NeuralForecast(models=[model], freq=train_args.freq)

    print("\n--- Starting Training ---")
    nf.fit(df=train_df, val_size=train_args.val_size)

    # 6. Make Predictions
    print("\n--- Generating Forecasts ---")
    forecasts_df = nf.predict(futr_df=test_df)
    forecasts_df = forecasts_df.reset_index()
    results_df = pd.merge(test_df, forecasts_df, on=['unique_id', 'ds'], how='left')

    # 7. Save Results
    forecast_csv_path = os.path.join(run_output_dir, 'forecasts.csv')
    results_df.to_csv(forecast_csv_path, index=False)
    print(f"Forecasts saved to: {forecast_csv_path}")

    # 8. Evaluate forecasts
    from utils import calculate_metrics, save_dict_to_json

    evaluation = calculate_metrics(results_df, model_name)
    print("\n--- Evaluation Metrics ---")
    for metric, value in evaluation.items():
        print(f"{metric.upper()}: {value:.4f}")

    summary = {
        'model_args': model_args,
        'evaluation': evaluation
    }
    summary_path = os.path.join(run_output_dir, 'summary.json')
    save_dict_to_json(summary, summary_path)
    print(f"\nEvaluation summary saved to: {summary_path}")

    # 8. Plot and Save Figure
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_df = results_df.set_index('ds')

    plot_df['y'].plot(ax=ax, label='Actual')
    plot_df[model_name].plot(ax=ax, label='Forecast')

    # If it's DistributionLoss, plot confidence intervals
    if f'{model_name}-lo-90' in plot_df.columns:
        ax.fill_between(
            plot_df.index,
            plot_df[f'{model_name}-lo-90'],
            plot_df[f'{model_name}-hi-90'],
            alpha=0.3,
            color='orange',
            label='90% Confidence Interval'
        )

    ax.set_title(f'{model_name} Forecast vs Actuals ({train_args.traffic_direction} traffic)')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Traffic (Mbps)')
    ax.legend()
    ax.grid(True)

    plot_path = os.path.join(run_output_dir, 'forecast_plot.png')
    fig.savefig(plot_path)
    print(f"Forecast plot saved to: {plot_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
