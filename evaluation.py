import argparse
import pandas as pd
import torch
from model import add_model_args
from neuralforecast.losses.pytorch import MAE, MAPE, MASE, MSE
from utils import load_and_process_data

def main():
    # Khởi tạo parser
    parser = argparse.ArgumentParser(description="Evaluate forecast performance using NeuralForecast metrics.")

    # Thêm các đối số cơ bản
    parser.add_argument('--forecast_path', type=str, required=True, help='Path to the input forecast file (.csv)')
    parser.add_argument('--training_path', type=str, required =True, help='Path to training data')

    # Thêm đối số model_name từ model.py (nếu chưa có)
    parser = add_model_args(parser)

    # Parse args
    args = parser.parse_args()

    # Đọc file CSV
    try:
        forecast_df = pd.read_csv(args.forecast_path)
        history_df = pd.read_excel(args.training_path)
    except Exception as e:
        print(f"[ERROR] Failed to read forecast/training file: {e}")
        return

    # Kiểm tra cột dự báo tồn tại
    pred_col = f'{args.model_name}-median' if f'{args.model_name}-median' in forecast_df.columns else args.model_name

    if pred_col not in forecast_df.columns:
        print(f"[ERROR] Column '{pred_col}' not found in forecast file.")
        print(f"Available columns: {forecast_df.columns.tolist()}")
        return

    # Lấy giá trị thật và dự báo
    y_true = torch.tensor(forecast_df['y'].values, dtype=torch.float32)
    y_pred = torch.tensor(forecast_df[pred_col].values, dtype=torch.float32)
    y_history_df, _ = load_and_process_data(file_path = args.training_path, train_scale = 1)
    y_history = torch.tensor(y_history_df['y'], dtype=torch.float32)

    # In kết quả các metric
    print(f"Model: {args.model_name}")
    print(f"Metrics:")
    print(f"  MAE : {MAE()(y_true, y_pred):.4f}")
    print(f"  MSE : {MSE()(y_true, y_pred, y_insample=y_history):.4f}")
    print(f"  MAPE: {MAPE()(y_true, y_pred, y_insample =y_history):.4f}")
   

if __name__ == "__main__":
    main()
