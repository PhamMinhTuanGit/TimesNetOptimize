# Time Series Forecasting with NeuralForecast

This project provides a flexible framework for training and evaluating state-of-the-art time series forecasting models from the `neuralforecast` library, including TimesNet, NHITS, and PatchTST. It is designed to be easily configurable through command-line arguments and shell scripts.

## Features

- **Modular Structure:** Code is organized into data processing, model definition, training, and inference.
- **Configurable Training:** Easily train different models with various hyperparameters using shell scripts or command-line arguments.
- **Comprehensive Evaluation:** Automatically calculates and saves key performance metrics (MAE, MSE, RMSE, MAPE).
- **Visualization:** Generates and saves plots of forecasts against actual values.
- **End-to-End Pipeline:** A single script (`full_pipeline.sh`) to run training, inference, evaluation, and visualization in one go.
- **Rolling Forecast Inference:** Includes a script to simulate real-world model performance using a rolling forecast methodology.

---

## Running on Google Colab

Follow these steps to set up and run this project in a Google Colab notebook.

### Step 1: Set up the Colab Environment

1.  Open a new notebook on Google Colab.
2.  **Mount Google Drive:** This allows you to access your project files.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  **Clone Your Repository:** Clone your GitHub repository into your Colab environment.
    ```bash
    !git clone https://github.com/PhamMinhTuanGit/TimesNetOptimize.git
    ```
4.  **Navigate to Project Directory:** Change the current working directory to your project folder.
    ```python
    import os
    os.chdir('TimesNetOptimize')
    ```

### Step 2: Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
!pip install -r requirements.txt
```

### Step 3: Upload Your Data

You need to upload your data file (`SLA0338SRT03_20250807114227010.xlsx`) to the `data/` directory.

1.  In the Colab file explorer on the left, navigate into the `TimesNetOptimize/` folder.
2.  Create a new folder named `data`.
3.  Right-click the `data` folder and select "Upload".
4.  Choose your `.xlsx` data file to upload it.

### Step 4: Train a Model

You can now run the training scripts directly from a Colab cell. The `!` at the beginning of the line tells Colab to execute it as a shell command.

**Example: Train the TimesNet model**

```bash
# Make the script executable first
!chmod +x ./train_timesnet.sh

# Run the training script
!./train_timesnet.sh
```

**Example: Train the NHITS model**

```bash
!chmod +x ./train_nhits.sh
!./train_nhits.sh
```

After training, you will find the results (logs, checkpoints, forecasts, and plots) in the `output/` directory.

### Step 5: Run Inference with a Trained Model

Once you have a trained model checkpoint, you can run inference.

1.  **Find your checkpoint directory:** After training, the path to the saved model directory will be printed (e.g., `output/TimesNet/TimesNet_2.1m`). Copy this path.
2.  **Run the inference pipeline script:**

```bash
# Make the script executable
!chmod +x ./inference.sh

# Run the full pipeline by providing the model name and the path to the checkpoint directory
!./inference.sh TimesNet "output/TimesNet/TimesNet_2.1m"
```

---

## Project Structure

```
├── data/                     # Data files
├── output/                   # Training outputs (logs, models, forecasts)
├── inference_output/         # Inference outputs
├── Data_processing.py        # Script for data loading and preprocessing
├── model.py                  # Model factory and argument definitions
├── training.py               # Main training script
├── inference.py              # Script for rolling forecast inference
├── utils.py                  # Utility functions (e.g., metrics)
├── train_timesnet.sh         # Example script to train TimesNet
├── train_nhits.sh            # Example script to train NHITS
├── inference.sh              # Example script to run inference
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```
