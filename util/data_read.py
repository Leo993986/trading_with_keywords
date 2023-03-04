import pandas as pd
import numpy as np

def read_csv_sorted(data_set_path):
    data_set = pd.read_csv(data_set_path)
    data_set = data_set.drop(['Adj Close'], axis=1)
    data_set['date'] = pd.to_datetime(data_set['Date'], format = '%Y-%m-%d')
    data_set['Date'] = data_set['date']
    data_set = data_set.drop(['date'], axis=1)
    data_set = data_set.sort_values(['Date'])

    return data_set