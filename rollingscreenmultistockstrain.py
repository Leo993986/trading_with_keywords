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

from stable_baselines.common.vec_env import  DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines import PPO2

from env.FintechMultiStocksEnv import FinTechMultiStocksTrainEnv
from util.technical_indicators import create_indicators_with_momentum, create_indicators
from util.data_read import read_csv_sorted

def rollingscreenmultistockstrain(data_set, algorithm, policy):
    gc.set_threshold(100, 5, 5)
    data_folder = 'data/'
    momentum_flag = 1
    # action_flag 0 -> 70% and 7.5% * 4
    # action_flag 1 -> -1 ~ +1
    # action_flag 2 -> action 0 ~ 1
    action_flag = 1
 
    data_set_choose = ["AGG2022", "VTI2022", "DBA2022", "VNQ2022", "GLD2022"]  

    base_folder = '/' + algorithm + '/' + policy + '/' + data_set + '/'
    tensorboard_folder = './tensorboard' + base_folder
    model_folder = './model' + base_folder
    debug_folder = './debug' + base_folder

    # 產生必要資料夾
    if not os.path.isdir(tensorboard_folder):
        os.makedirs(tensorboard_folder)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    if not os.path.isdir(debug_folder):
        os.makedirs(debug_folder)

    df_data_set_collection = [] 
    df_data_set = []  

    for data_set_name in data_set_choose:
        data_set_file_name = data_set_name + '.csv'
        data_set_path = data_folder + data_set_file_name
        df_data_set = read_csv_sorted(data_set_path)
        df_data_set_collection.append(df_data_set)

    # 設定訓練資料,評估資料,測試資料長度
    test_len = 252
    train_len = len(df_data_set) - test_len * 4
    validation_len =63
    trading_len = 63

    for i in range(16):
        # 切割訓練資料,評估資料,測試資料
        train_df_data_set = []
        for j in range(5):
            df_data_set_temp = df_data_set_collection[j]
            train_df_data_set_temp = df_data_set_temp[:train_len]
            train_df_data_set.append(train_df_data_set_temp)

        train_len = train_len + validation_len        

        # 添加技術指標
        for j in range(5):
            if momentum_flag == 1:
                train_df_data_set[j] = create_indicators_with_momentum(train_df_data_set[j].reset_index())
            else:
                train_df_data_set[j] = create_indicators(train_df_data_set[j].reset_index())
        
        train_df = [train_df_data_set[0], train_df_data_set[1], train_df_data_set[2], train_df_data_set[3], train_df_data_set[4]]
        
        # RL環境
        train_env = DummyVecEnv([lambda: FinTechMultiStocksTrainEnv(train_df, start_balance=10000, min_trading_unit=0, max_trading_count = 1000,max_change = 100, action_flag = action_flag, observation_length=int(3))])
        
        if algorithm == 'PPO2':
            model = PPO2(policy, train_env, verbose=0, nminibatches=1, tensorboard_log=tensorboard_folder + str(i) + '/', full_tensorboard_log=False)

        for idx in range(0, 100):
            model.learn(total_timesteps=len(train_df_data_set)*10)
            model.save("{}{}_{}".format(model_folder,str(i),str(idx)))
            gc.collect()

        del model
        del train_env
        del train_df
        del train_df_data_set

    del df_data_set_collection
    del df_data_set

    print('Finish: ' + algorithm + "-" + data_set)

if __name__ == '__main__':
    rollingscreenmultistockstrain('all', 'PPO2', 'MlpPolicy')