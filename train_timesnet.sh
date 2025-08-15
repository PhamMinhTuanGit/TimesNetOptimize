#!/bin/bash

# This script automates the process of training the TimesNet model
# using the main training script (training.py).

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting TimesNet training run..."

# --- Configuration ---
# Define variables for paths and hyperparameters to make the script easy to modify.
DATA_FILE="data/SLA0338SRT03_20250807114227010.xlsx"
TRAFFIC_DIRECTION="in"
OUTPUT_DIR="./output"

# Model-specific hyperparameters for this run
HORIZON=24
INPUT_SIZE=72
LOSS_FUNCTION="MAE" # Using MAE for a simple, robust loss
LEARNING_RATE=1e-4
MAX_STEPS=500
HIDDEN_SIZE=128
CONV_HIDDEN_SIZE=128

# --- Execute Training Command ---
# The backslashes (\) at the end of each line allow us to break the command
# into multiple lines for better readability.
python3 training.py \
    --data_path "${DATA_FILE}" \
    --traffic_direction "${TRAFFIC_DIRECTION}" \
    --output_dir "${OUTPUT_DIR}" \
    --freq "5min" \
    --val_size 24 \
    --model_name TimesNet \
    --h ${HORIZON} \
    --input_size ${INPUT_SIZE} \
    --loss ${LOSS_FUNCTION} \
    --learning_rate ${LEARNING_RATE} \
    --max_steps ${MAX_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --conv_hidden_size ${CONV_HIDDEN_SIZE}
   

echo "✅ TimesNet training run finished successfully!"
