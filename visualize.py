import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    """
    Loads forecast data from a CSV and generates a plot comparing
    actual values vs. predicted values over time.
    """
    parser = argparse.ArgumentParser(description="Visualize forecast results from a CSV file.")
    parser.add_argument('--forecast_path', type=str, required=True,
                        help='Path to the forecast CSV file. Must contain "ds", "y", and a prediction column.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model, used to identify the prediction column (e.g., "TimesNet").')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Directory to save the output plot.')
    args = parser.parse_args()

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(args.forecast_path)
        print(f"Successfully loaded forecast data from: {args.forecast_path}")
    except FileNotFoundError:
        print(f"[ERROR] File not found at: {args.forecast_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {e}")
        return

    # --- 2. Prepare Data for Plotting ---
    # Ensure 'ds' column is in datetime format
    if 'ds' not in df.columns:
        print("[ERROR] 'ds' column not found in the forecast file.")
        return
    df['ds'] = pd.to_datetime(df['ds'])

    # Identify prediction column
    # neuralforecast often names it {model_name} or {model_name}-median for point forecasts
    pred_col = args.model_name
    if pred_col not in df.columns:
        pred_col = f'{args.model_name}-median'
        if pred_col not in df.columns:
            print(f"[ERROR] Could not find prediction column '{args.model_name}' or '{pred_col}' in the file.")
            print(f"Available columns: {df.columns.tolist()}")
            return

    if 'y' not in df.columns:
        print("[ERROR] 'y' column (actual values) not found in the forecast file.")
        return

    # --- 3. Create and Save Plot ---
   
    fig, ax = plt.subplots(figsize=(15, 7))
    if pred_col == f'{args.model_name}-median':
        ax.plot(df['ds'], df['y'], label='Actual Values', color='blue', linestyle='-')
        ax.plot(df['ds'], df[pred_col], label=f'{args.model_name} Forecast', color='red', linestyle='--')
        ax.fill_between(df['ds'], df[f'{args.model_name}-hi-90'], df[f'{args.model_name}-lo-90'])
        ax.set_title(f'Forecast vs. Actuals for {args.model_name}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Traffic (Mbps)', fontsize=12)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        fig.tight_layout()
    else:
        pred_col == f'{args.model_name}'
        ax.plot(df['ds'], df['y'], label='Actual Values', color='blue', linestyle='-')
        ax.plot(df['ds'], df[pred_col], label=f'{args.model_name} Forecast', color='red', linestyle='--')
        ax.set_title(f'Forecast vs. Actuals for {args.model_name}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Traffic (Mbps)', fontsize=12)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.model_name}_forecast_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved successfully to: {output_path}")

if __name__ == '__main__':
    main()