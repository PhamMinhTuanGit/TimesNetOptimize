#!/bin/bash

# This script automates the process of running inference with a pre-trained model
# using the inference.py script. It performs a rolling forecast.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting rolling forecast inference..."

# --- Configuration ---
# IMPORTANT: You must update CHECKPOINT_PATH with the path to your trained model checkpoint.


MODEL_NAME="TimesNet"
CHECKPOINT_PATH=$1

DATA_FILE="data/SLA0338SRT03_20250807114227010.xlsx"
TRAFFIC_DIRECTION="out"
OUTPUT_DIR="./inference_output"
if [ $# -eq 0 ]; then
    echo "usage: ./inference.sh <CHECKPOINT_PATH>"
    exit 1
fi

# --- Execute Inference Command ---
python3 inference.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --data_path "${DATA_FILE}" \
    --traffic_direction "${TRAFFIC_DIRECTION}" \
    --output_dir "${OUTPUT_DIR}"

echo "✅ Inference run finished successfully!"
echo "Find your results in the '${OUTPUT_DIR}' directory."
