import pandas as pd
import numpy as np
import math
from scipy.stats import norm

def get_IPR(df, j):

  year_len = 252
  train_test_len = year_len * 5 -1
  obs_len = len(df)-train_test_len
  # 252*3-1=755
  train_test_df_data_set = df[obs_len:]
  train_test_df_data_set = train_test_df_data_set.reset_index()
  train_test_df_data_set = train_test_df_data_set.drop(['index'], axis=1)
  IPR = []
  df_data_set_week =[]
  # 755/5 = 151
  for i in range(5):
    df_data_set_week_d = df[i::5]
    df_data_set_week_d = df_data_set_week_d.reset_index()
    df_data_set_week_d = df_data_set_week_d.drop(['index'], axis=1)
    df_data_set_week.append(df_data_set_week_d)

  w = -1
  for t in range(obs_len, len(df)):
    # 假設3年151 + 當天 = 152
    weekly_return = []
    w+=1
    t_day = math.floor(t/5)+1

    for i in range(t_day-152,t_day-1):
      # print(str(i+1)+ " " + str(len(df_data_set_week[w%5]['Close'])) + " " + str(df_data_set_week[w%5]['Close'][i+1]))
      weekly_return.append((df_data_set_week[w%5]['Close'][i+1] - df_data_set_week[w%5]['Close'][i])/df_data_set_week[w%5]['Close'][i]) # 151筆
    
    #j = 13 # 13 26 39 52
    data_var = np.var(weekly_return)
    data_mean = np.mean(weekly_return)
    price_ln = math.log(df['Close'][t-1]/df['Close'][t - 1 - (j * 5)])
    IPR.append(norm.cdf((price_ln - j * data_mean)/math.sqrt(j*data_var)))
  
  train_test_df_data_set['IPR'] = IPR
  return train_test_df_data_set