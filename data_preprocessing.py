# data_preprocessing.py
import pandas as pd
import numpy as np


weather_df = pd.read_csv('data/weather_data.csv', parse_dates=['date'])
grocery_df = pd.read_csv('data/grocery_prices.csv', parse_dates=['date'])


merged_df = pd.merge(weather_df, grocery_df, on=['date', 'location'])


merged_df['precipitation'].fillna(0, inplace=True)


merged_df['temp_avg'] = (merged_df['temp_min'] + merged_df['temp_max']) / 2
merged_df['price_lag_7'] = merged_df.groupby('item')['price'].shift(7)  # 7-day lag


merged_df = pd.get_dummies(merged_df, columns=['weather_description'])


merged_df.to_csv('data/merged_data.csv', index=False)