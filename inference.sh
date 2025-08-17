#!/bin/bash

# This script automates running the full inference pipeline (inference, evaluation,
# and visualization) for a pre-trained model.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Argument Validation ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <MODEL_NAME> <CHECKPOINT_PATH>"
    echo "Example: $0 TimesNet 'output/TimesNet/TimesNet_2.1m'"
    exit 1
fi

MODEL_NAME=$1
CHECKPOINT_PATH=$2

if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint directory not found at '${CHECKPOINT_PATH}'"
    exit 1
fi

echo "🚀 Starting inference pipeline for model: ${MODEL_NAME}"

# --- 2. Configuration ---
# These can be customized if needed
DATA_FILE="data/SLA0338SRT03_20250807114227010.xlsx"
TRAFFIC_DIRECTION="out"
INFER_OUTPUT_DIR="./inference_output"
VIS_OUTPUT_DIR="./visualizations"

# --- 3. Run Inference ---
echo "🔍 Step 1/3: Running rolling forecast inference..."
# The inference script is run in silent mode to only output the final CSV path.
FORECAST_PATH=$(python3 inference.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --data_path "${DATA_FILE}" \
    --traffic_direction "${TRAFFIC_DIRECTION}" \
    --output_dir "${INFER_OUTPUT_DIR}" \
    --silent \
    | tail -n 1 | tr -d '\r')

echo "✅ Inference complete. Forecasts saved to: '${FORECAST_PATH}'"

# --- 4. Run Evaluation ---
echo "📊 Step 2/3: Evaluating forecast metrics..."
python3 evaluation.py \
    --forecast_path "${FORECAST_PATH}" \
    --training_path "${DATA_FILE}" \
    --model_name "${MODEL_NAME}"

# --- 5. Run Visualization ---
echo "🎨 Step 3/3: Generating forecast plot..."
python3 visualize.py \
    --forecast_path "${FORECAST_PATH}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${VIS_OUTPUT_DIR}"

echo "🎉 Inference pipeline finished successfully!"
