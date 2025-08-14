from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import DistributionLoss, MAE
from neuralforecast.losses.numpy import mse
import argparse



def get_time_series_model(args):
    time_step_out = args.time_step_out
    time_step_in = args.time_step_in
    hidden_size = args.hidden_size
    conv_hidden_size = args.conv_hidden_size
    lr = args.lr
    epochs = args.epochs
    freq = args.freq
    if args.loss == 'distribution':
        loss = DistributionLoss(distribution='Normal', level=[80, 90])
    elif args.loss == 'mae':
        loss = MAE()
    if args.model == "timesnet":
        model = TimesNet(
            h=time_step_out,
            input_size=time_step_in,
            hidden_size=hidden_size,
            conv_hidden_size=conv_hidden_size,
            loss=loss,
            scaler_type='standard',
            learning_rate=lr,
            max_steps=epochs,
            val_check_steps=50,
            early_stop_patience_steps=0
        )

        nf = NeuralForecast(
            models=[model],
            freq=freq
        )


