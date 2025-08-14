#!/bin/bash

# This script automates the process of training the NHITS model
# using the main training script (training.py).

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting NHITS training run..."

# --- Configuration ---
# Define variables for paths and hyperparameters to make the script easy to modify.
DATA_FILE="data/SLA0338SRT03_20250807114227010.xlsx"
TRAFFIC_DIRECTION="in"
OUTPUT_DIR="./output"

# Model-specific hyperparameters for this run
HORIZON=24
INPUT_SIZE=72
LOSS_FUNCTION="DistributionLoss" # DistributionLoss often works well with NHITS
LEARNING_RATE=1e-4
MAX_STEPS=1000
N_BLOCKS=3
MLP_UNITS_1=768
MLP_UNITS_2=768

# --- Execute Training Command ---
python /Users/phamminhtuan/Desktop/TimesNetOptimize/training.py \
    --data_path "${DATA_FILE}" \
    --traffic_direction "${TRAFFIC_DIRECTION}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name NHITS \
    --h ${HORIZON} \
    --input_size ${INPUT_SIZE} \
    --loss ${LOSS_FUNCTION} \
    --learning_rate ${LEARNING_RATE} \
    --max_steps ${MAX_STEPS} \
    --n_blocks ${N_BLOCKS} \
    --mlp_units ${MLP_UNITS_1} ${MLP_UNITS_2}

echo "✅ NHITS training run finished successfully!"
