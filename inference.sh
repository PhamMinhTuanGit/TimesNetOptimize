#!/bin/bash

# This script automates the process of running inference with a pre-trained model
# using the inference.py script. It performs a rolling forecast.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting rolling forecast inference..."

# --- Configuration ---
# IMPORTANT: You must update CHECKPOINT_PATH with the path to your trained model checkpoint.
# You can find it in the 'output' directory from a previous training run.
# Example: output/TimesNet/run_20231027-123456/checkpoints/best-checkpoint.ckpt
CHECKPOINT_PATH="PASTE_YOUR_CHECKPOINT_PATH_HERE"

MODEL_NAME="TimesNet" # Should match the model of the checkpoint
DATA_FILE="data/SLA0338SRT03_20250807114227010.xlsx"
TRAFFIC_DIRECTION="in"
OUTPUT_DIR="./inference_output"

# --- Validation ---
if [ "${CHECKPOINT_PATH}" == "PASTE_YOUR_CHECKPOINT_PATH_HERE" ]; then
    echo "❌ Error: Please update the CHECKPOINT_PATH variable in this script before running."
    exit 1
fi

# --- Execute Inference Command ---
python /Users/phamminhtuan/Desktop/TimesNetOptimize/inference.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --data_path "${DATA_FILE}" \
    --traffic_direction "${TRAFFIC_DIRECTION}" \
    --output_dir "${OUTPUT_DIR}"

echo "✅ Inference run finished successfully!"
echo "Find your results in the '${OUTPUT_DIR}' directory."
