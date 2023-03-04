import gym
import pandas as pd
import numpy as np
from sklearn import preprocessing
from empyrical import sortino_ratio, omega_ratio, sharpe_ratio

from util.data_processing import data_stationary

np.warnings.filterwarnings('ignore')

# PER_TRADE = 10
PER_TRADE = 50

class FinTechMultiStocksTrainEnv(gym.Env):
    '''Fintech 訓練環境'''
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self, df, start_balance = 10000, transaction_fees = 0, min_trading_unit = 5, max_trading_count = 10, max_change = 500, action_flag = 1, **kwargs):
        '''
        初始化環境

        : param df:                 (pandas.array)  歷史價格資料
        : param start_balance:      (int)           起始價格                            (預設: 10000元)
        : param transaction_fees:   (float)         手續費                              (預設: 0.002 = 0.2%) 
        : param min_trading_unit:   (int)           最小交易單位，小數點後幾位             (預設: 5)
        : param max_trading_count   (float)         最大交易數量(商品數量)                (預設: 10)
        : param max_change          (float)         最大漲跌幅(用於正規化資料集)           (預設: 500%)   
        '''

        self.start_balance = start_balance
        self.transaction_fees = transaction_fees
        self.min_trading_unit = min_trading_unit
        self.max_trading_count = max_trading_count
        self.max_change = max_change
        self.df_size = 0    
        # action_flag 0 -> 70% and 7.5% * 4
        # action_flag 1 -> -1 ~ +1
        # action_flag 2 -> action 0 ~ 1
        self.action_flag = action_flag

        self.df = []
        self.stationary_df = []
        
        # 處理NaN資料及平穩數據
        # https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
        for _df in df:
            temp_df = _df.fillna(method='bfill').reset_index(drop = True)
            self.df.append(temp_df)
            self.stationary_df.append(data_stationary(temp_df, ['Open', 'High', 'Low', 'Close', 'Volume']))
            self.df_size += 1

        # 正規化所有資料
        self.normalization()

        # 觀察歷史N個時間單位 預設10個時間單位
        self.observation_length = kwargs.get('observation_length', 10)
        
        # 5個股票的買賣
        if self.action_flag == 0:
            self.action_space = gym.spaces.MultiDiscrete([6])
        elif self.action_flag == 1:
            self.action_space = gym.spaces.Box(low = -1, high = 1,shape = (self.df_size,))
        elif self.action_flag == 2:
            self.action_space = gym.spaces.Box(low = 0, high = 1,shape = (self.df_size,))

        # 網路input的形狀
        # momentum ["AAPL", "F", "PG", "T", "XOM"] 
        # ["AGG", "VTI", "DBA", "VNQ", "GLD"] 
        # 用來交易的數據集(價差(最高,最低,開盤,收盤)+交易量差+技術指標-index-時間) 14
        # 自己的票倉(餘額　貨幣數量 購買數量 購買成本 賣出數量 賣出所得)
        # 多資產配置: 用來交易的數據集 * 5 + 餘額 + (持有數量+購買數量+購買成本+賣出數量+賣出所得) * 5
        # 共 96 個
        self.observation_shape = (1, (len(self.df[0].columns) - 2 ) * self.df_size + 1 + 5 * self.df_size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

    def reset(self):
        self.balance = self.start_balance
        self.cryptocurrency_held = [0] * self.df_size
        self.assets = [self.balance]     
        self.current_step = 0

        # 餘額　貨幣數量 購買數量 購買成本 賣出數量 賣出所得
        self.account_history = np.array([[self.balance]] + [[0]] * (5 * self.df_size))

        return self.next_observation()

    def step(self, actions):
        current_price = []
        next_time_price = []
        for i in range(self.df_size):
            current_price.append(self.df[i]['Close'].values[self.current_step + self.observation_length])
            next_time_price.append(self.df[i]['Close'].values[self.current_step + self.observation_length + 1])

        buy_amount = [0] * self.df_size
        cost = [0] * self.df_size
        sell_amount = [0] * self.df_size
        income = [0] * self.df_size
        mean_price = 0

        if self.action_flag == 0:
            choose_percentage = [0.075] * 5
            # 股票AAPL 70% 其餘各7.5% 0
            # 股票F 70% 其餘各7.5% 1
            # 股票PG 70% 其餘各7.5% 2
            # 股票T 70% 其餘各7.5% 3
            # 股票XOM 70% 其餘各7.5% 4
            choose_percentage[actions[0]] = 0.7
            buy_amount, cost, sell_amount, income = self.buy_sell_with_percentage(current_price, choose_percentage)

        elif self.action_flag == 1:
            # action = 1 的狀況下
            # 每個 current_price 乘上 X 盡可能等於一個定值
            # 設mean_price為定值 ， X = (mid_price//current_price[i])
            # mid_price = np.percentile(current_price, 50) * PER_TRADE
            mean_price = np.mean(current_price, axis=0) * PER_TRADE

            for i in range(self.df_size):
                    actions[i] *= (mean_price//current_price[i])

            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                if self.cryptocurrency_held[index] > 0:
                    sell_amount[index] = min(abs(actions[index]), self.cryptocurrency_held[index], self.max_trading_count)
                    sell_amount[index] = int(sell_amount[index] * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
                    income[index] = (1 - self.transaction_fees) * sell_amount[index] * current_price[index]

                    self.balance += income[index]
                    self.cryptocurrency_held[index] -= sell_amount[index]

            for index in buy_index:
                amount = self.balance // current_price[index]

                buy_amount[index] = min(actions[index], amount, self.max_trading_count)
                buy_amount[index] = int(buy_amount[index] * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
                cost[index] = buy_amount[index] * current_price[index] * ( 1 + self.transaction_fees)

                self.balance -= cost[index]
                self.cryptocurrency_held[index] += buy_amount[index]

        elif self.action_flag == 2:
            choose_percentage = [0] * 5
            action_sum = sum(actions)
            if action_sum != 0:
                for i in range(self.df_size):
                    choose_percentage[i] = actions[i] / action_sum

            buy_amount, cost, sell_amount, income = self.buy_sell_with_percentage(current_price, choose_percentage)


        current_history = [[self.balance]]
        for i in range(self.df_size):
            current_history += [[self.cryptocurrency_held[i]], [buy_amount[i]], [cost[i]], [sell_amount[i]], [income[i]]] 

        self.account_history = np.append(self.account_history,current_history, axis=1)

        self.current_step += 1
        
        next_time_asset = 0
        for i in range(self.df_size):
            next_time_asset += self.cryptocurrency_held[i] * next_time_price[i]
        next_time_asset += self.balance
        self.assets.append(next_time_asset)

        observation = self.next_observation()
        reward = self.get_reward()
        finish = self.finish()

        return observation, reward, finish, {'asset': self.assets[len(self.assets) - 2], 'current_price': current_price}

    def buy_sell_with_percentage(self,current_price, percentagechoose):
        asset = []
        balance = 0
        cryptocurrency_held = []
        cryptocurrency_held = self.cryptocurrency_held
        balance = self.balance

        current_asset = 0
        for i in range(self.df_size):
            asset.append(cryptocurrency_held[i] * current_price[i])
            current_asset += cryptocurrency_held[i] * current_price[i]
        current_asset += balance

        buy_amount = [0] * self.df_size
        cost = [0] * self.df_size
        sell_amount = [0] * self.df_size
        income = [0] * self.df_size

        
        for i in range(self.df_size):
            percentage = percentagechoose[i]

            if asset[i]/current_asset > percentage:
                amount = ((asset[i] - percentage * current_asset)/current_price[i])/cryptocurrency_held[i]
                sell_amount[i] = cryptocurrency_held[i] * amount if cryptocurrency_held[i] * amount <= self.max_trading_count else self.max_trading_count * amount
                # 處理最小交易單位
                sell_amount[i] = int(sell_amount[i] * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
                income[i] = (1 - self.transaction_fees) * sell_amount[i] * current_price[i]

                balance += income[i]
                cryptocurrency_held[i] -= sell_amount[i]
        
        for i in range(self.df_size):
            percentage = percentagechoose[i]

            if asset[i]/current_asset < percentage:
                cost[i] = percentage * current_asset - asset[i] if percentage * current_asset - asset[i] <= balance else balance
                buy_amount[i] = (1 - self.transaction_fees) * cost[i] / current_price[i] if (1 - self.transaction_fees) * cost[i] / current_price[i] <= self.max_trading_count else self.max_trading_count
                # 處理最小交易單位
                buy_amount[i] = int(buy_amount[i] * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
                # 重新計算成本
                cost[i] = buy_amount[i] * current_price[i] * ( 1 + self.transaction_fees)

                balance -= cost[i]
                cryptocurrency_held[i] += buy_amount[i]     

        self.cryptocurrency_held = cryptocurrency_held
        self.balance = balance

        return buy_amount, cost, sell_amount, income

    def render(self, mode='human'):
        price_str = ""
        amount_str = ""
        for i in range(self.df_size):
            price_str += "," + str(self.df[i]['Close'].values[self.current_step + self.observation_length])
            amount_str += "," + str(self.account_history[1 + i*5][-1]) + \
                          "," + str(self.account_history[2 + i*5][-1]) + \
                          "," + str(self.account_history[4 + i*5][-1])
        
        print(price_str + ',' + str(self.account_history[0][-1]) + amount_str + ',' + str(self.assets[-1]))

    def finish(self):
        return self.assets[-1] < self.start_balance / 10 or self.current_step + self.observation_length == len(self.df[0]) - 1

    def get_reward(self):
        length = min(self.current_step, self.observation_length)
        assets = self.assets[-2:]

        # 計算收益率 
        # (assets[n] - assets[n -1])/assets[n -1]
        income_difference = np.diff(assets)
        returns = income_difference / assets[:len(assets) - 1]

        # 過濾雜訊(極微小的漲跌幅)
        returns = np.round(returns, 5)

        if np.count_nonzero(returns) < 1:
            return 0

        reward = returns[0] * 100
        return reward if np.isfinite(reward) else 0

    def next_observation(self):
        observation = []
        # 交易資料集的內容
        scaler = preprocessing.MinMaxScaler()
        obs_data = self.scaled_df[0][:self.current_step + self.observation_length]
        observation = obs_data.values[-1]

        # 觀察資料集的內容 len(self.scaled_df) -> 1 評估 
        for index in range(1,len(self.scaled_df)):
            obs_data = self.scaled_df[index][:self.current_step + self.observation_length]
            observation = np.hstack((observation,obs_data.values[-1]))

        # 加入這一次交易資訊
        scaled_history = scaler.fit_transform(self.account_history.astype('float32'))
        observation = np.insert(observation, len(observation), scaled_history[:, -1], axis=0) # length +1 +5*5
        
        observation[np.bitwise_not(np.isfinite(observation))] = 0

        observation = [observation]
        return observation

    def normalization(self):
        scaler = preprocessing.MinMaxScaler()

        self.scaled_df = []

        for stationary_df in self.stationary_df:
            stationary_data = stationary_df
            # 抓出除了 Index和Date其他的資料
            features = stationary_data[stationary_data.columns.difference(['index', 'Date'])]
            # 將無窮大的資料變成0
            scaled_data = features.values
            scaled_data[abs(scaled_data) == np.inf] = 0
            features = pd.DataFrame(scaled_data, columns=features.columns)

            # 將(open High low close)一起正規化
            scaler_max = self.max_change
            scaler_min = -self.max_change

            scaled_data = features[['Open', 'High', 'Low', 'Close']].values
            scaled_data = (scaled_data - scaler_min) / (scaler_max - scaler_min)
            features[['Open', 'High', 'Low', 'Close']] = pd.DataFrame(scaled_data, columns=features[['Open', 'High', 'Low', 'Close']].columns)

            # 其他的資料各別正規化
            scaled_data = features[features.columns.difference(['Open', 'High', 'Low', 'Close'])].values
            scaled_data = scaler.fit_transform(scaled_data.astype('float32'))
            features[features.columns.difference(['Open', 'High', 'Low', 'Close'])] = pd.DataFrame(scaled_data, columns=features[features.columns.difference(['Open', 'High', 'Low', 'Close'])].columns)

            self.scaled_df.append(features)
