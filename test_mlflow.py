import mlflow
import mlflow.prophet
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from utils import load_and_process_data
# Load sample time series data (Prophet expects 'ds' and 'y' columns)
# This example uses the classic Peyton Manning Wikipedia page views dataset

df, _ = load_and_process_data('data/T56.xlsx', traffic_direction='out', train_scale=1)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
print(f"Data preview:\n{df.head()}")

with mlflow.start_run(run_name="Basic Prophet Forecast"):
    # Create Prophet model with specific parameters
    model = Prophet(
        changepoint_prior_scale=0.05,  # Flexibility of trend changes
        seasonality_prior_scale=10,  # Strength of seasonality
        holidays_prior_scale=10,  # Strength of holiday effects
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    # Fit the model
    model.fit(df)

    # Extract and log model parameters
    def extract_prophet_params(prophet_model):
        """Extract Prophet model parameters for logging."""
        from prophet.serialize import SIMPLE_ATTRIBUTES

        params = {}
        for attr in SIMPLE_ATTRIBUTES:
            if hasattr(prophet_model, attr):
                value = getattr(prophet_model, attr)
                if isinstance(value, (int, float, str, bool)):
                    params[attr] = value
        return params

    params = extract_prophet_params(model)
    mlflow.log_params(params)

    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=365)  # Forecast 1 year ahead
    forecast = model.predict(future)

    # Cross-validation for model evaluation
    cv_results = cross_validation(
        model,
        initial="730 days",  # Initial training period
        period="180 days",  # Spacing between cutoff dates
        horizon="365 days",  # Forecast horizon
        parallel="threads",  # Use threading for speed
    )

    # Calculate performance metrics
    metrics = performance_metrics(cv_results)
    avg_metrics = metrics[["mse", "rmse", "mae", "mape"]].mean().to_dict()
    mlflow.log_metrics(avg_metrics)

    # Log the model with input example
    mlflow.prophet.log_model(
        pr_model=model, name="prophet_model", input_example=df[["ds"]].head(10)
    )

    print(f"Model trained and logged successfully!")
    print(f"Average MAPE: {avg_metrics['mape']:.2f}%")