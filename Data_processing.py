import pandas as pd
import numpy as np

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



# traffic_in_clipped = clip_upper_outliers(traffic_in)
# traffic_out_clipped = clip_upper_outliers(traffic_out)
# df = df.rename(columns={'dataTime':'timestamp'})
# # df.to_csv('traffic_in.csv')
# print(df.head())
# df = pd.read_csv('traffic_in.csv')
# df = df.rename(columns={'KpiDataFindResult.trafficInStr': 'y', 'timestamp': 'ds'})
# df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y %H:%M:%S')
# df['unique_id'] = 'series_1'
# df = df[['unique_id', 'ds', 'y']]

# total_len = len(df)
# train_len = int(total_len * 0.8)
# Y_train_df = df.iloc[:train_len].reset_index(drop=True)
# Y_test_df = df.iloc[train_len:].reset_index(drop=True)
def get_traffic_in_df(df: pd.DataFrame):
    traffic_in_df = pd.DataFrame()
    traffic_in_df['y']=df['KpiDataFindResult.trafficInStr']
    traffic_in_df['y']=traffic_in_df['y'].apply(convert_speed_to_mbps)
    traffic_in_df['ds']=pd.to_datetime(df['dataTime'], format='%d/%m/%Y %H:%M:%S')
    traffic_in_df['unique_id'] = 'series_1'
    return traffic_in_df
def get_traffic_out_df(df:pd.DataFrame):
    traffic_out_df = pd.DataFrame()
    traffic_out_df['y']=df['KpiDataFindResult.trafficOutStr']
    traffic_out_df['y']=traffic_out_df['y'].apply(convert_speed_to_mbps)
    traffic_out_df['ds']=pd.to_datetime(df['dataTime'], format='%d/%m/%Y %H:%M:%S')
    traffic_out_df['unique_id'] = 'series_1'
    return traffic_out_df


if __name__ == '__main__':
    df = pd.read_excel('data/SLA0338SRT03_20250807114227010.xlsx',  header = 3)

    df  = df.drop(columns =['No.', 'Device name', 'Device IP', 'Interface name', 'speed', 'unit',
       'interfaceId','KpiDataFindResult.packetErrorStr',
       'KpiDataFindResult.trafficInRaw', 'KpiDataFindResult.trafficOutRaw',
       'KpiDataFindResult.packetErrorRaw', 'fromTime', 'toTime'], )
    traffic_out_df = get_traffic_out_df(df)
    traffic_in_df = get_traffic_in_df(df)
    print(traffic_in_df.head(5), traffic_out_df.head(5))