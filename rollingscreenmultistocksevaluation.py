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
from stable_baselines import PPO2, A2C, ACKTR

from env.FintechMultiStocksEnv import FinTechMultiStocksTrainEnv
from util.technical_indicators import create_indicators_with_momentum, create_indicators
from util.sharpe_ratio import sharpe_ratio
from util.data_read import read_csv_sorted

def run_agent(env, model, TNX_df, test_amount):
    obs = env.reset()
    done, assets, total_reword, _states, asset = False, [], 0, None, 0
    week_assets, week_count = [], 0

    while not done:
        action, _states = model.predict(obs, _states)
        if test_amount > 0:
            for i in range(test_amount):
                if i != 0:
                    print(',', end = '')
                print(str(action[0][i]), end = '')
            env.render()
        obs, reward, done, info = env.step(action)

        asset = info[0]['asset']

        assets.append(asset)
        current_price = info[0]['current_price']

        week_count = week_count + 1
        if done or (week_count % 5) == 0:
            week_assets.append(asset)

    rate_of_returns = []
    for week_asset in week_assets:
        rate_of_returns.append((week_asset - 10000) / 10000 * 100)
    rate_of_return = (asset - 10000) / 10000 * 100

    assets = pd.DataFrame(assets, columns=['Close'], dtype=float)
    TNX_df = TNX_df[4:]
    TNX_df = TNX_df.reset_index()
    assets['TNX'] = TNX_df['Close']
    # sharpe, profit, variance = sharpe_ratio(assets, 4)

    #為了計算4年的夏普值
    sharpe = assets

    del model
    del env

    return rate_of_return, sharpe, rate_of_returns

def run_test(df, idx, model_folder, i, TNX_df, test_flag, action_flag):
    env = DummyVecEnv([lambda: FinTechMultiStocksTrainEnv(df, start_balance=10000, min_trading_unit=0, max_trading_count = 1000,max_change = 100, action_flag = action_flag, observation_length=int(3))])
    
    model = PPO2.load("{}{}_{}".format(model_folder,str(i),str(idx)), env)

    if test_flag == 1:
        rate_of_return, sharpe, rate_of_returns = run_agent(env, model, TNX_df, len(df))
    else:
        rate_of_return, sharpe, rate_of_returns = run_agent(env, model, TNX_df, 0)
    return rate_of_return, sharpe, rate_of_returns

def run_evaluation(df, model_folder, i, TNX_df, action_flag):
    env = DummyVecEnv([lambda: FinTechMultiStocksTrainEnv(df, start_balance=10000, min_trading_unit=0, max_trading_count = 1000,max_change = 100, action_flag = action_flag, observation_length=int(3))])

    all_rate_of_return = []

    for idx in range(0, 100):
        if not os.path.exists("{}{}_{}".format(model_folder,str(i),str(idx)) + '.zip'):
            break
        model = PPO2.load("{}{}_{}".format(model_folder,str(i),str(idx)), env)

        rate_of_return, sharpe, rate_of_returns = run_agent(env, model, TNX_df, 0)
        all_rate_of_return.append(rate_of_return)

    argmaxs = np.argsort(all_rate_of_return)[::-1][:10]

    return argmaxs

def compute_benchmark(df, cryptocurrency_held, balance = 10000, buy = 1):
    df_size = len(df)
    start_price = []
    end_price = [[] for j in range(12)]
    transaction_fees = 0
    max_trading_count = 1000
    min_trading_unit = 0
    observation_length = 3

    for i in range(df_size):
        start_price.append(df[i]['Close'].values[observation_length])
        for j in range(11):
            end_price[j].append(df[i]['Close'].values[observation_length - 1 + (j + 1) * 5])
        end_price[11].append(df[i]['Close'].values[len(df[0]) - 1])

    if buy == 1:
        current_asset = balance
        for i in range(df_size):
            percentage = 0.2
            cost = percentage * current_asset if percentage * current_asset <= balance else balance
            buy_amount = (1 - transaction_fees) * cost / start_price[i] if (1 - transaction_fees) * cost / start_price[i] <= max_trading_count else max_trading_count
            # 處理最小交易單位
            buy_amount = int(buy_amount * pow(10, min_trading_unit))/pow(10, min_trading_unit)

            # 重新計算成本
            cost = buy_amount * start_price[i] * ( 1 + transaction_fees)

            balance -= cost
            cryptocurrency_held[i] += buy_amount   

    rate_of_returns = []
    for j in range(12):
        asset = 0
        for i in range(df_size):
            asset += cryptocurrency_held[i] * end_price[j][i]
        asset += balance
        rate_of_returns.append((asset - 10000) / 10000 * 100)

    # rate_of_return = (asset - 10000) / 10000 * 100

    return balance, cryptocurrency_held, rate_of_returns

def rollingscreenmultistocksevaluation(data_set, algorithm, policy):
    return_list = []
    return_list.append(data_set)
    gc.set_threshold(100, 5, 5)
    data_folder = 'data/'
    momentum_flag = 1
    test_flag = 0
    sharpe4years = 1
    # action_flag 0 -> 70% and 7.5% * 4
    # action_flag 1 -> -1 ~ +1
    # action_flag 2 -> action 0 ~ 1
    action_flag = 1

    TNX_name = "TNX2022"
    data_set_choose = ["AGG2022", "VTI2022", "DBA2022", "VNQ2022", "GLD2022"] 

    TNX_data_set_path = data_folder + TNX_name + '.csv'

    base_folder = '/' + algorithm + '/' + policy + '/' + data_set + '/'
    tensorboard_folder = './tensorboard' + base_folder
    model_folder = './model' + base_folder
    debug_folder = './debug' + base_folder

    df_data_set_collection = [] 
    df_data_set = []  

    for data_set_name in data_set_choose:
        data_set_file_name = data_set_name + '.csv'
        data_set_path = data_folder + data_set_file_name
        df_data_set = read_csv_sorted(data_set_path)
        df_data_set_collection.append(df_data_set)


    TNX_data_set = read_csv_sorted(TNX_data_set_path)
    
    # 切割訓練資料,評估資料,測試資料
    test_len = 252
    train_len = len(df_data_set) - test_len * 4
    validation_len =63
    trading_len = 63

    balance, cryptocurrency_held = 10000, [0] * len(data_set_choose)
    
    if sharpe4years == 1:
        if test_flag == 1:
            assets_collection = pd.DataFrame([], columns=['Close', 'TNX'], dtype=float)
        else:
            assets_collection = [pd.DataFrame([], columns=['Close', 'TNX'], dtype=float)] * 100

    best_model_idxs_collection = ''

    if test_flag == 1:
        action_str = ""
        price_str = ""
        amount_str = ""
        for i in range(len(data_set_choose)):
            action_str += "action" + str(i+1) if i == 0 else ",action" + str(i+1)
            price_str += ",Price" + str(i+1)
            amount_str += ",cryptocurrency held" + str(i+1) + ",buy amount" + str(i+1) + ",sell amount" + str(i+1)
        print(action_str + price_str + ",balance" + amount_str + ",Net Worth")


    for i in range(16):
        train_df_data_set = []
        validation_df_data_set = []
        trading_df_data_set = []
        for j in range(5):
            df_data_set_temp = df_data_set_collection[j]
            train_df_data_set_temp = df_data_set_temp[:train_len]
            validation_df_data_set_temp = train_df_data_set_temp[(len(train_df_data_set_temp) - trading_len):]
            trading_df_data_set_temp = df_data_set_temp[len(train_df_data_set_temp):(len(train_df_data_set_temp) + trading_len)]
            train_df_data_set.append(train_df_data_set_temp)
            validation_df_data_set.append(validation_df_data_set_temp)
            trading_df_data_set.append(trading_df_data_set_temp)

        train_TNX_data_set = TNX_data_set[:train_len]
        validation_TNX_data_set = train_TNX_data_set[(len(train_TNX_data_set) - trading_len):]
        trading_TNX_data_set = TNX_data_set[len(train_TNX_data_set):(len(train_TNX_data_set) + trading_len)]

        train_len = train_len + validation_len        

        # 添加技術指標
        for j in range(5):
            if momentum_flag == 1:
                train_df_data_set[j] = create_indicators_with_momentum(train_df_data_set[j].reset_index())
                validation_df_data_set[j] = create_indicators_with_momentum(validation_df_data_set[j].reset_index())
                trading_df_data_set[j] = create_indicators_with_momentum(trading_df_data_set[j].reset_index())
            else:
                train_df_data_set[j] = create_indicators(train_df_data_set[j].reset_index())
                validation_df_data_set[j] = create_indicators(validation_df_data_set[j].reset_index())
                trading_df_data_set[j] = create_indicators(trading_df_data_set[j].reset_index())
        
        train_df = [train_df_data_set[0], train_df_data_set[1], train_df_data_set[2], train_df_data_set[3], train_df_data_set[4]]
        validation_df = [validation_df_data_set[0], validation_df_data_set[1], validation_df_data_set[2], validation_df_data_set[3], validation_df_data_set[4]]
        trading_df = [trading_df_data_set[0], trading_df_data_set[1], trading_df_data_set[2], trading_df_data_set[3], trading_df_data_set[4]]

        trading_result = []
        trading_benchmark_result = []
        if sharpe4years == 0:
            trading_sharpe = []
        week_trading_result = [[] for j in range(12)]

        best_model_idxs = run_evaluation(validation_df, model_folder, i, validation_TNX_data_set, action_flag)

        if test_flag == 1:
            for idx in best_model_idxs:
                best_model_idxs_collection += str(i) + '_' + str(idx)
                if i < 15: best_model_idxs_collection += ','
                rate_of_return, sharpe, rate_of_returns = run_test(trading_df, idx, model_folder, i, trading_TNX_data_set, test_flag, action_flag)
                if sharpe4years == 1:
                    assets_collection = assets_collection.append(sharpe, ignore_index=True) 
                break
        else:
            k = -1
            for idx in best_model_idxs:
                best_model_idxs_collection += str(i) + '_' + str(idx)
                if (i < 15 or k < 9): best_model_idxs_collection += ','
                k += 1
                for j in range(10):
                    rate_of_return, sharpe, rate_of_returns = run_test(trading_df, idx, model_folder, i, trading_TNX_data_set, test_flag, action_flag)
                    trading_result.append(rate_of_return)
                    for n in range(12):
                        week_trading_result[n].append(rate_of_returns[n])
                    if sharpe4years == 0:
                        trading_sharpe.append(sharpe)
                    else:
                        assets_collection[10*k+j] = assets_collection[10*k+j].append(sharpe, ignore_index=True)


            # # 開場5個ETF各20%並以3個月為交易時長
            rate_of_benchmark_return = []
            if (i % 1) == 0:
                cryptocurrency_held = [0] * len(data_set_choose)
                balance, cryptocurrency_held, rate_of_benchmark_return = compute_benchmark(trading_df, cryptocurrency_held)
            else:
                balance, cryptocurrency_held, rate_of_benchmark_return = compute_benchmark(trading_df, cryptocurrency_held, balance, 0)
            trading_benchmark_result = rate_of_benchmark_return


            for n in range(12):
                return_list.append(np.mean(week_trading_result[n], axis=0))
                return_list.append(trading_benchmark_result[n])                
            # return_list.append(np.mean(trading_result, axis=0))
            # return_list.append(np.mean(trading_benchmark_result, axis=0))
            if sharpe4years == 0:
                return_list.append(np.mean(trading_sharpe, axis=0))

        if sharpe4years == 0:
            del trading_sharpe
        del trading_result
        del trading_benchmark_result

        del train_df
        del validation_df
        del trading_df
        
        del train_df_data_set
        del validation_df_data_set
        del trading_df_data_set

    if sharpe4years == 1:
        if test_flag == 1:
            sharpe, profit, variance = sharpe_ratio(assets_collection, 4)
        else:
            sharpe_collection = []
            for i in range(100):
                sharpe, profit, variance = sharpe_ratio(assets_collection[i], 4)
                sharpe_collection.append(sharpe)
            return_list.append(np.mean(sharpe_collection, axis=0))

            del sharpe_collection
        del assets_collection

    del df_data_set_collection
    del df_data_set
    del TNX_data_set

    print(best_model_idxs_collection)

    if test_flag == 1 and sharpe4years == 1:
        print('Finish: ' + algorithm + "-" + data_set + ' sharpe : ' + str(sharpe))
    else:
        print('Finish: ' + algorithm + "-" + data_set)

    return return_list

if __name__ == '__main__':
    rollingscreenmultistocksevaluation('all', 'PPO2', 'MlpPolicy')