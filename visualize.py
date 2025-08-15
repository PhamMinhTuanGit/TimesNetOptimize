import pandas as pd
import torch
import matplotlib.pyplot as plt

df = pd.read_csv('inference_output/TimesNet_20250815-080713/rolling_forecast.csv')
plt.figure(figsize=[16,8])
plt.plot(df['TimesNet'], color = 'r', label = 'prediction')
plt.plot(df['y'], color = 'b', label = 'Truth')
plt.legend()
plt.savefig('test.png')