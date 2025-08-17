#!/bin/bash

# Exit if any command fails
set -e

echo "🚀 Starting TimesNet training run..."

# --- Configuration ---
DATA_FILE="data/SLA0338SRT03_20250807114925121.xlsx"
TRAFFIC_DIRECTION="out"
OUTPUT_DIR="./output"

MODEL_NAME="TimesNet"
HORIZON=24
INPUT_SIZE=72
LOSS_FUNCTION="MAPE"
LEARNING_RATE=1e-4
MAX_STEPS=100
HIDDEN_SIZE=144
CONV_HIDDEN_SIZE=128

# --- Run Training ---
CHECKPOINT_PATH=$(python3 training.py \
    --data_path "${DATA_FILE}" \
    --traffic_direction "${TRAFFIC_DIRECTION}" \
    --output_dir "${OUTPUT_DIR}" \
    --freq "5min" \
    --val_size 0 \
    --model_name "${MODEL_NAME}" \
    --h ${HORIZON} \
    --input_size ${INPUT_SIZE} \
    --loss ${LOSS_FUNCTION} \
    --learning_rate ${LEARNING_RATE} \
    --max_steps ${MAX_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --conv_hidden_size ${CONV_HIDDEN_SIZE} \
    | tail -n 1| tr -d '\r')

echo "✅ Training finished!"
echo "🔖 Saved checkpoint: $CHECKPOINT_PATH"

# --- Inference Configuration ---
echo "🔍 Starting inference..."
INFER_DATA_FILE="data/SLA0338SRT03_20250807114227010.xlsx"
INFER_OUTPUT_DIR="./inference_output"

FORECAST_PATH=$(python3 inference.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --data_path "${INFER_DATA_FILE}" \
    --traffic_direction "${TRAFFIC_DIRECTION}" \
    --output_dir "${INFER_OUTPUT_DIR}"
    | tail -n 1 | tr -d '\r')

echo "✅ Inference run finished successfully!"
echo "📁 Forecasts saved to: '${FORECAST_PATH}'"

# --- Run Evaluation ---
echo "📊 Starting evaluation..."
python3 evaluation.py \
    --forecast_path "${FORECAST_PATH}" \
    --training_path "${DATA_FILE}" \
    --model_name "${MODEL_NAME}"

echo "📊 Evaluation completed successfully!"

# --- Run Visualization ---
echo "🎨 Generating forecast plot..."
python3 visualize.py \
    --forecast_path "${FORECAST_PATH}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "./visualizations"

echo "🖼️  Visualization complete!"