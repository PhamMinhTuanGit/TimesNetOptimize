import pandas as pd

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

def process_traffic_data(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    Processes raw DataFrame to extract and format traffic data for NeuralForecast.
    
    Args:
        df (pd.DataFrame): The raw DataFrame from the Excel file.
        direction (str): The traffic direction, either 'in' or 'out'.
        
    Returns:
        pd.DataFrame: A formatted DataFrame with 'unique_id', 'ds', and 'y' columns.
    """
    if direction not in ['in', 'out']:
        raise ValueError("Direction must be 'in' or 'out'")
    
    col_name = f'KpiDataFindResult.traffic{direction.capitalize()}Str'
    
    processed_df = pd.DataFrame()
    processed_df['y'] = df[col_name].apply(convert_speed_to_mbps)
    processed_df['ds'] = pd.to_datetime(df['dataTime'], format='%d/%m/%Y %H:%M:%S')
    processed_df['unique_id'] = 'series_1'
    
    return processed_df

if __name__ == '__main__':
    df = pd.read_excel('data/SLA0338SRT03_20250807114227010.xlsx',  header = 0)
    # Clean column names to remove leading/trailing whitespace
    df.columns = df.columns.str.strip()
    traffic_out_df = process_traffic_data(df, direction='out')
    traffic_in_df = process_traffic_data(df, direction='in')
    print(traffic_in_df.head(5), traffic_out_df.head(5))