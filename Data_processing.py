import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast.models import TimesNet
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE



df = pd.read_excel('data/SLA0338SRT03_20250807114227010.xlsx',  header = 3)

df  = df.drop(columns =['No.', 'Device name', 'Device IP', 'Interface name', 'speed', 'unit',
       'interfaceId','KpiDataFindResult.packetErrorStr',
       'KpiDataFindResult.trafficInRaw', 'KpiDataFindResult.trafficOutRaw',
       'KpiDataFindResult.packetErrorRaw', 'fromTime', 'toTime'], )

def convert_speed_to_mbps(speed_str):
    value, unit = speed_str.split()
    value = float(value)
    unit = unit.lower()
    if unit == 'gbps':
        return value * 1000
    elif unit == 'mbps':
        return value
    else:
        return None


def parse_bandwidth(bandwidth_str):
    value, unit = bandwidth_str.split()
    value = float(value)
    if unit.lower() == 'Gbps':
        mbps = value * 1000
        
        return mbps
    return None
def clip_upper_outliers(series):
    """
    Clips values in a pandas Series that are above the upper whisker (Q3 + 1.5*IQR)
    to the maximum non-outlier value.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 2 * iqr

    # Find max value within the non-outlier range
    max_non_outlier = series[series <= upper_bound].max()

    # Clip values above the upper bound
    return series.apply(lambda x: min(x, max_non_outlier))

df['KpiDataFindResult.trafficInStr'] = df['KpiDataFindResult.trafficInStr'].apply(convert_speed_to_mbps)
df['KpiDataFindResult.trafficOutStr'] = df['KpiDataFindResult.trafficOutStr'].apply(convert_speed_to_mbps)
traffic_in = df['KpiDataFindResult.trafficInStr']
traffic_out = df['KpiDataFindResult.trafficOutStr']
traffic_in_clipped = clip_upper_outliers(traffic_in)
traffic_out_clipped = clip_upper_outliers(traffic_out)
df = df.drop(columns=['KpiDataFindResult.trafficOutStr'])
df = df.rename(columns={'dataTime':'timestamp'})
df.to_csv('traffic_in.csv')
print(df.head())