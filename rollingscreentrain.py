import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import gym
import pandas as pd
import numpy as np
import gc
import random


from stable_baselines.common.vec_env import  DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines import PPO2

from env.FintechEnv import FinTechTrainEnv
from env.FintechBondEnv import FinTechBondTrainEnv

from util.technical_indicators import create_indicators, create_indicators_with_momentum
from util.data_read import read_csv_sorted


def rollingscreentrain(data_set, algorithm, policy):
    gc.set_threshold(100, 5, 5)
    data_folder = 'data/'
    bond_flag = 0
    momentum_flag = 1

    bond_name = "IEF2022"
    data_set += "2022"

    data_set_file_name = data_set + '.csv'
    data_set_path = data_folder + data_set_file_name

    bond_data_set_path = data_folder + bond_name + '.csv'

    base_folder = '/' + algorithm + '/' + policy + '/' + data_set + '/'
    tensorboard_folder = './tensorboard' + base_folder
    model_folder = './model' + base_folder
    debug_folder = './debug' + base_folder
    trendc = pd.read_csv('data/AXPtrend.csv')
    trendc['Date'] = pd.to_datetime(trendc['date'])
    
    trendc = trendc.drop(['date'], axis=1)
    
   #trende = pd.read_csv('data/trendc.csv')
    #trende['Date'] = pd.to_datetime(trende['date'])
    #trende = trende.drop(['date'], axis=1)


    # 產生必要資料夾
    if not os.path.isdir(tensorboard_folder):
        os.makedirs(tensorboard_folder)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    if not os.path.isdir(debug_folder):
        os.makedirs(debug_folder)

    df_data_set = read_csv_sorted(data_set_path)

    if bond_flag == 1:
        bond_data_set = read_csv_sorted(bond_data_set_path)
    
    # 設定訓練資料,評估資料,測試資料長度
    test_len = 252
    train_len = len(df_data_set) - test_len * 4
    validation_len =63
    trading_len = 63

    for i in range(16):
        # 切割訓練資料,評估資料,測試資料
        train_df_data_set = df_data_set[:train_len]
        
        if bond_flag == 1: 
            train_bond_data_set = bond_data_set[:train_len]
            
        train_len = train_len + validation_len        
        train_df_data_set = pd.merge(train_df_data_set, trendc)
        #train_df_data_set = pd.merge(train_df_data_set, trende)

        # 添加技術指標
        if momentum_flag == 1:
            train_df_data_set = create_indicators_with_momentum(train_df_data_set.reset_index())
        else:
            train_df_data_set = create_indicators(train_df_data_set.reset_index())
            
        if bond_flag == 0: 
            train_df = [train_df_data_set]
        else:
            train_df = [train_df_data_set, train_bond_data_set]
        

            
        # RL環境
        if bond_flag == 0: 
            train_env = DummyVecEnv([lambda: FinTechTrainEnv(train_df, start_balance=10000, min_trading_unit=0, max_trading_count=1000,max_change=100, observation_length=int(3))])
        else:
            train_env = DummyVecEnv([lambda: FinTechBondTrainEnv(train_df, start_balance=10000, min_trading_unit=0, max_trading_count=1000,max_change=100, observation_length=int(3))])

        model = PPO2(policy, train_env, verbose=0, nminibatches=1, tensorboard_log=tensorboard_folder + str(i) + '/', full_tensorboard_log=False)

        for idx in range(0, 100):
            # model.learn(total_timesteps=len(train_df_data_set)*10)
            model.learn(total_timesteps=20000)
            model.save("{}{}_{}".format(model_folder,str(i),str(idx)))
            gc.collect()
      
        del model
        del train_env
        del train_df
        del train_df_data_set

        if bond_flag == 1: 
            del train_bond_data_set

    del df_data_set
    
    if bond_flag == 1: 
        del bond_data_set

    print('Finish: ' + algorithm + "-" + data_set)
    

if __name__ == '__main__':
    rollingscreentrain('AXP', 'PPO2', 'MlpPolicy')
