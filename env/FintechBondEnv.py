import gym
import pandas as pd
import numpy as np
from sklearn import preprocessing
from empyrical import sortino_ratio, omega_ratio, sharpe_ratio

from util.data_processing import data_stationary

np.warnings.filterwarnings('ignore')

class FinTechBondTrainEnv(gym.Env):
    '''Fintech 訓練環境'''
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self, df, start_balance = 10000, transaction_fees = 0, min_trading_unit = 5, max_trading_count = 10, max_change = 500, **kwargs):
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

        self.df = []
        self.stationary_df = []
        
        # 處理NaN資料及平穩數據
        # https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
        for _df in df:
            temp_df = _df.fillna(method='bfill').reset_index(drop = True)
            self.df.append(temp_df)
            self.stationary_df.append(data_stationary(temp_df, ['Open', 'High', 'Low', 'Close', 'Volume']))

        # 正規化所有資料
        self.normalization()

        # 觀察歷史N個時間單位 預設10個時間單位
        self.observation_length = kwargs.get('observation_length', 10)
        
        # 三個動作(買 賣 不動作)  
        self.action_space = gym.spaces.MultiDiscrete([6])

        # 網路input的形狀
        # 用來交易的數據集(價差(最高,最低,開盤,收盤)+交易量差+技術指標-index-時間) 15
        # 其它觀察的數據集(價差(最高,最低,開盤,收盤)) 0
        # 自己的票倉(餘額　貨幣數量 購買數量 購買成本 賣出數量 賣出所得)
        self.observation_shape = (1, len(self.df[0].columns) - 2 + (len(self.df) - 1) * 4 + 1 + 5 * 2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

    def reset(self):
        self.balance = self.start_balance
        self.cryptocurrency_held = 0
        self.assets = [self.balance]     
        self.current_step = 0

        self.bond_cryptocurrency_held = 0

        # 餘額　貨幣數量 購買數量 購買成本 賣出數量 賣出所得
        self.account_history = np.array([[self.balance],[self.cryptocurrency_held],[self.bond_cryptocurrency_held],[0],[0],[0],[0],[0],[0],[0],[0]])

        return self.next_observation()

    def step(self, action):
        current_price = self.df[0]['Close'].values[self.current_step + self.observation_length]
        next_time_price = self.df[0]['Close'].values[self.current_step + self.observation_length + 1]

        current_bond_price = self.df[1]['Close'].values[self.current_step + self.observation_length]
        next_time_bond_price = self.df[1]['Close'].values[self.current_step + self.observation_length + 1]        

        buy_amount = 0
        cost = 0
        sell_amount = 0
        income = 0
        bond_buy_amount = 0
        bond_cost = 0
        bond_sell_amount = 0
        bond_income = 0
        current_signal_line = self.signal_line

        # 100%股票
        if action[0] == 0:
            if self.bond_cryptocurrency_held != 0:
                bond_sell_amount = self.bond_cryptocurrency_held if self.bond_cryptocurrency_held <= self.max_trading_count else self.max_trading_count
                # 處理最小交易單位
                bond_sell_amount = int(bond_sell_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
                bond_income = (1 - self.transaction_fees) * bond_sell_amount * current_bond_price

                self.balance += bond_income
                self.bond_cryptocurrency_held -= bond_sell_amount

            cost = self.balance
            buy_amount = (1 - self.transaction_fees) * cost / current_price if (1 - self.transaction_fees) * cost / current_price <= self.max_trading_count else self.max_trading_count
            # 處理最小交易單位
            buy_amount = int(buy_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
            # 重新計算成本
            cost = buy_amount * current_price * ( 1 + self.transaction_fees)

            self.balance -= cost
            self.cryptocurrency_held += buy_amount

        # 70%股票、30%債券
        elif action[0] == 1:
            buy_amount, cost, sell_amount, income, bond_buy_amount, bond_cost, bond_sell_amount, bond_income = self.buy_sell_with_percentage(0.7, 0.3, current_price, current_bond_price)

        # 50%股票、50%債券
        elif action[0] == 2:
            buy_amount, cost, sell_amount, income, bond_buy_amount, bond_cost, bond_sell_amount, bond_income = self.buy_sell_with_percentage(0.5, 0.5, current_price, current_bond_price)

        # 30%股票、70%債券
        elif action[0] == 3:
            buy_amount, cost, sell_amount, income, bond_buy_amount, bond_cost, bond_sell_amount, bond_income = self.buy_sell_with_percentage(0.3, 0.7, current_price, current_bond_price)

        # 100%債券
        elif action[0] == 4:
            if self.cryptocurrency_held != 0:
                sell_amount = self.cryptocurrency_held if self.cryptocurrency_held <= self.max_trading_count else self.max_trading_count
                # 處理最小交易單位
                sell_amount = int(sell_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
                income = (1 - self.transaction_fees) * sell_amount * current_price

                self.balance += income
                self.cryptocurrency_held -= sell_amount

            bond_cost = self.balance
            bond_buy_amount = (1 - self.transaction_fees) * bond_cost / current_bond_price if (1 - self.transaction_fees) * bond_cost / current_bond_price <= self.max_trading_count else self.max_trading_count
            # 處理最小交易單位
            bond_buy_amount = int(bond_buy_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
            # 重新計算成本
            bond_cost = bond_buy_amount * current_bond_price * ( 1 + self.transaction_fees)

            self.balance -= bond_cost
            self.bond_cryptocurrency_held += bond_buy_amount

        self.account_history = np.append(self.account_history,[[self.balance], [self.cryptocurrency_held], [self.bond_cryptocurrency_held], [buy_amount], [cost], [sell_amount], [income], [bond_buy_amount], [bond_cost], [bond_sell_amount], [bond_income]], axis=1)

        self.current_step += 1

        self.assets.append(self.balance + self.cryptocurrency_held * next_time_price + self.bond_cryptocurrency_held * next_time_bond_price)

        observation = self.next_observation()
        reward = self.get_reward()
        finish = self.finish()

        return observation, reward, finish, {'asset': self.assets[len(self.assets) - 2], 'current_price': current_price}

    def buy_sell_with_percentage(self, percentage, bond_percentage, current_price, current_bond_price):
        asset = self.cryptocurrency_held * current_price
        bond_asset = self.bond_cryptocurrency_held * current_bond_price
        current_asset = self.balance + bond_asset + asset

        buy_amount = 0
        cost = 0
        sell_amount = 0
        income = 0
        bond_buy_amount = 0
        bond_cost = 0
        bond_sell_amount = 0
        bond_income = 0

        if bond_asset/current_asset < bond_percentage and asset/current_asset < percentage:
            amount = percentage - asset/current_asset

            cost = current_asset * amount if current_asset * amount < self.balance else self.balance
            buy_amount = (1 - self.transaction_fees) * cost / current_price if (1 - self.transaction_fees) * cost / current_price <= self.max_trading_count else self.max_trading_count
            # 處理最小交易單位
            buy_amount = int(buy_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
            # 重新計算成本
            cost = buy_amount * current_price * ( 1 + self.transaction_fees)

            self.balance -= cost
            self.cryptocurrency_held += buy_amount

            bond_cost = self.balance
            bond_buy_amount = (1 - self.transaction_fees) * bond_cost / current_bond_price if (1 - self.transaction_fees) * bond_cost / current_bond_price <= self.max_trading_count else self.max_trading_count
            # 處理最小交易單位
            bond_buy_amount = int(bond_buy_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
            # 重新計算成本
            bond_cost = bond_buy_amount * current_bond_price * ( 1 + self.transaction_fees)

            self.balance -= bond_cost
            self.bond_cryptocurrency_held += bond_buy_amount

        elif bond_asset/current_asset > bond_percentage and asset/current_asset < percentage:
            amount = ((bond_asset - bond_percentage * current_asset)/current_bond_price)/self.bond_cryptocurrency_held
            bond_sell_amount = self.bond_cryptocurrency_held * amount if self.bond_cryptocurrency_held * amount <= self.max_trading_count else self.max_trading_count * amount
            # 處理最小交易單位
            bond_sell_amount = int(bond_sell_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
            bond_income = (1 - self.transaction_fees) * bond_sell_amount * current_bond_price

            self.balance += bond_income
            self.bond_cryptocurrency_held -= bond_sell_amount

            cost = self.balance
            buy_amount = (1 - self.transaction_fees) * cost / current_price if (1 - self.transaction_fees) * cost / current_price <= self.max_trading_count else self.max_trading_count
            # 處理最小交易單位
            buy_amount = int(buy_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
            # 重新計算成本
            cost = buy_amount * current_price * ( 1 + self.transaction_fees)

            self.balance -= cost
            self.cryptocurrency_held += buy_amount

        elif bond_asset/current_asset < bond_percentage and asset/current_asset > percentage:
            amount = ((asset - percentage * current_asset)/current_price)/self.cryptocurrency_held
            sell_amount = self.cryptocurrency_held * amount if self.cryptocurrency_held * amount <= self.max_trading_count else self.max_trading_count * amount
            # 處理最小交易單位
            sell_amount = int(sell_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
            income = (1 - self.transaction_fees) * sell_amount * current_price

            self.balance += income
            self.cryptocurrency_held -= sell_amount

            bond_cost = self.balance
            bond_buy_amount = (1 - self.transaction_fees) * bond_cost / current_bond_price if (1 - self.transaction_fees) * bond_cost / current_bond_price <= self.max_trading_count else self.max_trading_count
            # 處理最小交易單位
            bond_buy_amount = int(bond_buy_amount * pow(10, self.min_trading_unit))/pow(10, self.min_trading_unit)
            # 重新計算成本
            bond_cost = bond_buy_amount * current_bond_price * ( 1 + self.transaction_fees)

            self.balance -= bond_cost
            self.bond_cryptocurrency_held += bond_buy_amount

        return buy_amount, cost, sell_amount, income, bond_buy_amount, bond_cost, bond_sell_amount, bond_income

    def render(self, mode='human'):
        print(',' + str(self.df[0]['Close'].values[self.current_step + self.observation_length]) +
              ',' + str(self.df[1]['Close'].values[self.current_step + self.observation_length]) + 
              ',' + str(self.account_history[0][-1]) +
              ',' + str(self.account_history[1][-1]) +
              ',' + str(self.account_history[2][-1]) +
              ',' + str(self.account_history[3][-1]) +
              ',' + str(self.account_history[5][-1]) +
              ',' + str(self.account_history[7][-1]) +
              ',' + str(self.account_history[9][-1]) +
              ',' + str(self.assets[-1]))

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

    def computational_signal_line(self):
        signal_line = [1, 1, 1]
        if self.balance < self.df[0]['Close'].values[self.current_step + self.observation_length]:
            signal_line[0] = 0
        
        if self.cryptocurrency_held == 0:
            signal_line[1] = 0
            signal_line[2] = 0
        elif self.cryptocurrency_held / 5 < 1:
            signal_line[1] = 0

        self.signal_line = signal_line

    def next_observation(self):
        self.computational_signal_line()

        # 交易資料集的內容
        scaler = preprocessing.MinMaxScaler()
        obs_data = self.scaled_df[0][:self.current_step + self.observation_length]
        observation = obs_data.values[-1]

        # 觀察資料集的內容 len(self.scaled_df) -> 1 評估 沒進去
        for index in range(1,len(self.scaled_df)):
            obs_data = self.scaled_df[index][:self.current_step + self.observation_length]
            observation = np.hstack((observation,obs_data[['Open', 'High', 'Low', 'Close']].values[-1]))

        # 加入這一次交易資訊
        scaled_history = scaler.fit_transform(self.account_history.astype('float32'))
        observation = np.insert(observation, len(observation), scaled_history[:, -1], axis=0) # length +6 +5
        
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
