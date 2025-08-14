import pandas as pd
import matplotlib.pyplot as plt
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import DistributionLoss
df = pd.read_csv('traffic_in.csv')
df = df.rename(columns={'KpiDataFindResult.trafficInStr': 'y', 'timestamp': 'ds'})
df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y %H:%M:%S')
df['unique_id'] = 'series_1'
df = df[['unique_id', 'ds', 'y']]

total_len = len(df)
train_len = int(total_len * 0.8)
Y_train_df = df.iloc[:train_len].reset_index(drop=True)
Y_test_df = df.iloc[train_len:].reset_index(drop=True)

if torch.cuda.is_available():
    print("CUDA is available! Training on GPU...")
    accelerator = 'gpu'
    devices = [0]
else:
    print("CUDA is not available. Training on CPU.")
    accelerator = 'cpu'
    devices = 'auto'

h_forecast = 24
total_steps_to_predict = len(Y_test_df)

model = TimesNet(
    h=h_forecast,
    input_size=72,
    hidden_size=144,
    conv_hidden_size=144,
    loss=DistributionLoss(distribution='Normal', level=[80, 90]),
    scaler_type='standard',
    learning_rate=1e-3,
    max_steps=100,
    val_check_steps=50,
    early_stop_patience_steps=0
)

nf = NeuralForecast(
    models=[model],
    freq='5min'
)

nf.fit(
    df=Y_train_df,
    val_size=0,

)

nf.save(path='checkpoints_cpu/timesnet_model/')

