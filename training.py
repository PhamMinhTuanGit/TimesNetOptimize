import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import DistributionLoss
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
from Data_processing import get_traffic_in_df, get_traffic_out_df



df = pd.read_excel('data/SLA0338SRT03_20250807114227010.xlsx',  header = 3)
df  = df.drop(columns =['No.', 'Device name', 'Device IP', 'Interface name', 'speed', 'unit',
    'interfaceId','KpiDataFindResult.packetErrorStr',
    'KpiDataFindResult.trafficInRaw', 'KpiDataFindResult.trafficOutRaw',
    'KpiDataFindResult.packetErrorRaw', 'fromTime', 'toTime'], )
total_len = len(df)
train_len = int(total_len * 0.8)
Y_train_df = df.iloc[:train_len].reset_index(drop=True)
Y_test_df = df.iloc[train_len:].reset_index(drop=True)

logger = TensorBoardLogger("lightning_logs", name="TimesNet")
version_dir = os.path.join(logger.save_dir, logger.name, f"version_{logger.version}")
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(version_dir, "checkpoints"),
    filename="best-checkpoint",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)


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
    early_stop_patience_steps=2,
    callbacks=[checkpoint_callback],  # thêm callback vào đây
    logger=logger
)

nf = NeuralForecast(
    models=[model],
    freq='5min'
)

nf.fit(
    df=Y_train_df,
    val_size=12,

)



