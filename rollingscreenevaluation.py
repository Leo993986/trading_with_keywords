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

from env.FintechEnv import FinTechTrainEnv
from env.FintechBondEnv import FinTechBondTrainEnv
from util.technical_indicators import create_indicators, create_indicators_with_momentum
from util.sharpe_ratio import sharpe_ratio
from util.data_read import read_csv_sorted

def run_agent(env, model, TNX_df, test, bond_flag):
    obs = env.reset()
    done, assets, total_reword, _states = False, [], 0, None
    week_assets, week_count = [], 0

    while not done:
        action, _states = model.predict(obs, _states)
        if test == True:
            print(str(action[0][0]), end = '')
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

def run_test(df, idx, model_folder, i, TNX_df, bond_flag, test_flag):
    if bond_flag == 0:
        env = DummyVecEnv([lambda: FinTechTrainEnv(df, start_balance=10000, min_trading_unit=0, max_trading_count = 1000,max_change = 100, observation_length=int(3))])
    else:
        env = DummyVecEnv([lambda: FinTechBondTrainEnv(df, start_balance=10000, min_trading_unit=0, max_trading_count = 1000,max_change = 100, observation_length=int(3))])

    model = PPO2.load("{}{}_{}".format(model_folder,str(i),str(idx)), env)

    rate_of_return, sharpe, rate_of_returns = run_agent(env, model, TNX_df, test_flag, bond_flag)

    return rate_of_return, sharpe, rate_of_returns

def run_evaluation(df, model_folder, i, TNX_df, bond_flag):
    if bond_flag == 0:
        env = DummyVecEnv([lambda: FinTechTrainEnv(df, start_balance=10000, min_trading_unit=0, max_trading_count = 1000,max_change = 100, observation_length=int(3))])
    else:
        env = DummyVecEnv([lambda: FinTechBondTrainEnv(df, start_balance=10000, min_trading_unit=0, max_trading_count = 1000,max_change = 100, observation_length=int(3))])

    all_rate_of_return = []

    for idx in range(0, 100):
        if not os.path.exists("{}{}_{}".format(model_folder,str(i),str(idx)) + '.zip'):
            break
        model = PPO2.load("{}{}_{}".format(model_folder,str(i),str(idx)), env)

        rate_of_return, sharpe, rate_of_returns = run_agent(env, model, TNX_df, False, bond_flag)
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
        percentage = 0.5
        for i in range(df_size):
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

def rollingscreenevaluation(data_set, algorithm, policy):
    return_list = []
    return_list.append(data_set)
    gc.set_threshold(100, 5, 5)
    data_folder = 'data/'
    bond_flag = 0
    momentum_flag = 1
    test_flag = False
    sharpe4years = 1

    TNX_name = "TNX2022"
    bond_name = "IEF2022"
    data_set += "2022"

    data_set_file_name = data_set + '.csv'
    data_set_path = data_folder + data_set_file_name
    
    TNX_data_set_path = data_folder + TNX_name + '.csv'

    bond_data_set_path = data_folder + bond_name + '.csv'

    base_folder = '/' + algorithm + '/' + policy + '/' + data_set + '/'
    tensorboard_folder = './tensorboard' + base_folder
    model_folder = './model' + base_folder
    debug_folder = './debug' + base_folder

    df_data_set = read_csv_sorted(data_set_path)

    TNX_data_set = read_csv_sorted(TNX_data_set_path)
    trendc = pd.read_csv('data/AXPtrend.csv')
    trendc['Date'] = pd.to_datetime(trendc['date'])
    trendc = trendc.drop(['date'], axis=1)
    
    #trende = pd.read_csv('data/trendc.csv')
    #trende['Date'] = pd.to_datetime(trende['date'])
    #trende = trende.drop(['date'], axis=1)

    if bond_flag == 1:
        bond_data_set = read_csv_sorted(bond_data_set_path)

    # 切割訓練資料,評估資料,測試資料
    test_len = 252
    train_len = len(df_data_set) - test_len * 4
    validation_len =63
    trading_len = 63

    balance = 10000
    if bond_flag == 1:
        cryptocurrency_held = [0] * 2
    else:
        cryptocurrency_held = [0]

    if sharpe4years == 1:
        if test_flag:
            assets_collection = pd.DataFrame([], columns=['Close', 'TNX'], dtype=float)
        else:
            assets_collection = [pd.DataFrame([], columns=['Close', 'TNX'], dtype=float)] * 100

    if test_flag:
        if bond_flag == 1:
            print("action,Price,BondPrice,balance,cryptocurrency held,Bond cryptocurrency held,buy amount,sell amount,Bond buy amount,Bond sell amount,Net Worth")
        else:
            print("action,Price,balance,cryptocurrency held,buy amount,sell amount,Net Worth")
    
    
    best_model_idxs_collection = ''

    for i in range(16):
        train_df_data_set = df_data_set[:train_len]
        validation_df_data_set = train_df_data_set[(len(train_df_data_set) - trading_len):]
        trading_df_data_set = df_data_set[len(train_df_data_set):(len(train_df_data_set) + trading_len)]

        train_TNX_data_set = TNX_data_set[:train_len]
        validation_TNX_data_set = train_TNX_data_set[(len(train_TNX_data_set) - trading_len):]
        trading_TNX_data_set = TNX_data_set[len(train_TNX_data_set):(len(train_TNX_data_set) + trading_len)]


        if bond_flag == 1: 
            train_bond_data_set = bond_data_set[:train_len]
            validation_bond_data_set = train_bond_data_set[(len(train_bond_data_set) - trading_len):]
            trading_bond_data_set = bond_data_set[len(train_bond_data_set):(len(train_bond_data_set) + trading_len)]

        train_len = train_len + validation_len        
        train_df_data_set = pd.merge(train_df_data_set, trendc)
        validation_df_data_set = pd.merge(validation_df_data_set, trendc)
        trading_df_data_set = pd.merge(trading_df_data_set, trendc)
        #train_df_data_set = pd.merge(train_df_data_set, trende)
        #validation_df_data_set = pd.merge(validation_df_data_set, trende)
        #trading_df_data_set = pd.merge(trading_df_data_set, trende)
        # 添加技術指標
        if momentum_flag == 1:
            train_df_data_set = create_indicators_with_momentum(train_df_data_set.reset_index())
            validation_df_data_set = create_indicators_with_momentum(validation_df_data_set.reset_index())
            trading_df_data_set = create_indicators_with_momentum(trading_df_data_set.reset_index())
        else:
            train_df_data_set = create_indicators(train_df_data_set.reset_index())
            validation_df_data_set = create_indicators(validation_df_data_set.reset_index())
            trading_df_data_set = create_indicators(trading_df_data_set.reset_index())

        if bond_flag == 0: 
            train_df = [train_df_data_set]
            validation_df = [validation_df_data_set]
            trading_df = [trading_df_data_set]
        else:
            train_df = [train_df_data_set, train_bond_data_set]
            validation_df = [validation_df_data_set, validation_bond_data_set]
            trading_df = [trading_df_data_set, trading_bond_data_set]
        



        trading_result = [[] for j in range(10)]

        trading_benchmark_result = []
        if sharpe4years == 0:
            trading_sharpe = []
        week_trading_result = [[] for j in range(12)]

        best_model_idxs = run_evaluation(validation_df, model_folder, i, validation_TNX_data_set, bond_flag) 

        if test_flag:
            for idx in best_model_idxs:
                best_model_idxs_collection += str(i)+ '_' + str(idx)
                if(i < 15): best_model_idxs_collection += ','
                rate_of_return, sharpe, rate_of_returns = run_test(trading_df, idx, model_folder, i, trading_TNX_data_set, bond_flag, test_flag)
                if sharpe4years == 1:
                    assets_collection = assets_collection.append(sharpe, ignore_index=True) 
                break
        else:
            k = -1
            for idx in best_model_idxs:
                best_model_idxs_collection += str(i)+ '_' + str(idx)
                if(i < 15 or k < 9): best_model_idxs_collection += ','
                k += 1
                for j in range(10):
                    rate_of_return, sharpe, rate_of_returns = run_test(trading_df, idx, model_folder, i, trading_TNX_data_set, bond_flag, test_flag)
                    for n in range(12):
                        week_trading_result[n].append(rate_of_returns[n])
                    trading_result[j].append(rate_of_return)
                    if sharpe4years == 0:
                        trading_sharpe.append(sharpe)
                    else:
                        assets_collection[10*k+j] = assets_collection[10*k+j].append(sharpe, ignore_index=True)
                    
            # 開場現金股票或現金債券各20%並以3個月為交易時長
            rate_of_benchmark_return = 0
            if (i % 1) == 0:
                if bond_flag == 1:
                    cryptocurrency_held = [0] * 2
                else:
                    cryptocurrency_held = [0]
                balance, cryptocurrency_held, rate_of_benchmark_return = compute_benchmark(trading_df, cryptocurrency_held)
            else:
                balance, cryptocurrency_held, rate_of_benchmark_return = compute_benchmark(trading_df, cryptocurrency_held, balance, 0)
            trading_benchmark_result = rate_of_benchmark_return

            for n in range(10):
                #return_list.append(np.mean(week_trading_result[n], axis=0))
                return_list.append(np.mean(trading_result[n], axis=0))
                #return_list.append(trading_benchmark_result[n])   
                
                #return_list.append(np.mean(trading_benchmark_result, axis=0))
            if sharpe4years == 0:
                return_list.append(np.mean(trading_sharpe, axis=0)) 

        del trading_result
        if sharpe4years == 0:
            del trading_sharpe

        del train_df
        del validation_df
        del trading_df

        del train_df_data_set
        del validation_df_data_set
        del trading_df_data_set
        
        if bond_flag == 1: 
            del train_bond_data_set
            del validation_bond_data_set
            del trading_bond_data_set

    if sharpe4years == 1:
        if test_flag:
            total_change = []
            sharpe, profit, variance ,pct_change_Close = sharpe_ratio(assets_collection, 4)
            total_change.append(pct_change_Close)

        else:
            sharpe_collection = []
            total_change = []
            for i in range(100):
                sharpe, profit, variance, pct_change_Close = sharpe_ratio(assets_collection[i], 4)
                total_change.append(pct_change_Close)
                sharpe_collection.append(sharpe)
            return_list.append(np.mean(sharpe_collection, axis=0))


    print(best_model_idxs_collection)

    del df_data_set
    del TNX_data_set
    
    if bond_flag == 1: 
        del bond_data_set
        
    if test_flag and sharpe4years == 1:
        print('Finish: ' + algorithm + "-" + data_set + ' sharpe : ' + str(sharpe))
        print("asset :" + str(total_change))
    else:
        print('Finish: ' + algorithm + "-" + data_set)

    return return_list 
if __name__ == '__main__':
    rollingscreenevaluation('AXP', 'PPO2', 'MlpPolicy')