#!/bin/bash

# This script automates the process of running inference with a pre-trained model
# using the inference.py script. It performs a rolling forecast.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting rolling forecast inference..."

# --- Configuration ---
# IMPORTANT: You must update CHECKPOINT_PATH with the path to your trained model checkpoint.

if [ "$#" -ne 2 ]; then
    echo "❌ Usage: $0 <MODEL_NAME> <PATH_TO_CHECKPOINT>"
    echo "   Example: $0 TimesNet output/TimesNet/run_.../checkpoints/best-checkpoint.ckpt"
    exit 1
fi

MODEL_NAME=$1
CHECKPOINT_PATH=$2

DATA_FILE="data/SLA0338SRT03_20250807114227010.xlsx"
TRAFFIC_DIRECTION="in"
OUTPUT_DIR="./inference_output"

# --- Execute Inference Command ---
python /Users/phamminhtuan/Desktop/TimesNetOptimize/inference.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --data_path "${DATA_FILE}" \
    --traffic_direction "${TRAFFIC_DIRECTION}" \
    --output_dir "${OUTPUT_DIR}"

echo "✅ Inference run finished successfully!"
echo "Find your results in the '${OUTPUT_DIR}' directory."
