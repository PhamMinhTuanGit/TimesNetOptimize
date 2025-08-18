#!/bin/bash
set -e
echo ""
MODEL_NAME = $1
FORECAST_PATH = $2
DATA_FILE = $3


echo "📊 Starting evaluation..."
python3 evaluation.py \
    --forecast_path "${FORECAST_PATH}" \
    --training_path "${DATA_FILE}" \
    --model_name "${MODEL_NAME}"

echo "📊 Evaluation completed successfully!"