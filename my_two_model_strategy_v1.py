# 2024.6.17
# load two trained models to predict min(buy) and max(sell)
# daily trade with actual data and simulate client 

import warnings
warnings.filterwarnings('ignore')
import time
import datetime
from datetime import datetime, timedelta
from pytz import timezone
import logging
import smtplib
import joblib
import yfinance as yf
import akshare as ak
import pandas as pd
import numpy as np
import pandas_ta as pd_ta

from ta.trend import SMAIndicator, EMAIndicator, CCIIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.linear_model import LinearRegression

# import ta
# import matplotlib.pyplot as plt
# import plotly.graph_objs as go
# import seaborn as sns 
# import quantstats as qs

# import sklearn
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn import tree
# from imblearn.over_sampling import SMOTE

# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# from scipy import linalg
# from scipy.signal import argrelextrema
# import math
# from collections import OrderedDict
# from tqdm import tqdm

# ----------------------------------

class Client:
    def __init__(self, cash):        
        self.cash = cash 
        self.holds = {} # {stock:[shares, buy_price, cost, max_price], }
        self.history = [] # [[date, buy/sell, stock, price, shares], ]
        self.log = []
        self.total = cash
        
    def buy(self, stock, price, shares):
        content = f'Buy: {stock}, at {price}, for {shares}'
        send_email(content, content)
        if shares == 0 or self.cash < price * shares: return
        self.cash -= price * shares
        if stock not in self.holds: 
            self.holds[stock] = [shares, price, price*shares, price]
        else: 
            self.holds[stock][0] += shares
            self.holds[stock][2] += price*shares # cost 
            self.holds[stock][1] = self.holds[stock][2] / self.holds[stock][0]
        self.history.append([today, 'buy', stock, price, shares])
        
    def sell(self, stock, price, shares=None):
        content = f'Sell: {stock}, at {price}, for {shares}'
        send_email(content, content)
        if shares == 0 or self.holds.get(stock, None) is None: return
        if shares is None or shares >= self.holds[stock][0]: 
            shares = self.holds[stock][0]
        self.cash += price * shares
        if self.holds[stock][0] == shares: 
            self.holds.pop(stock)
        else: 
            self.holds[stock][0] -= shares
            self.holds[stock][2] -= price*shares # cost
            self.holds[stock][1] = self.holds[stock][2] / self.holds[stock][0]
        self.history.append([today, 'sell', stock, price, shares])        
        
    def get_info(self):
        return self.cash, self.holds

    def get_stock(self, stock):
        return self.holds.get(stock, None)
        
    def daily_update(self):
        if today.isoweekday() in [6, 7]: return
        content = f'==== {today} ====\nHolds: {self.holds}\nCash: {self.cash}\nTotal: {self.total}\n\n'
        send_email(f'{today} Log', content)
        logger.debug(f'==== {today} ====')
        logger.debug(f'== Holds:')
        total, close = 0.0, False
        for stock, data in self.holds.items():
            try:
                value = data[0] * get_stock_current_price(stock) # data[0]: shares                
                total += value
            except:
                value = data[2]
                close = True
            logger.debug(f'{stock}, {data}, |, {value}, {value-data[2]}') # data[2]: cost
        logger.debug(f'== Cash: {self.cash}')
        if not close:
            self.log.append([str(today), total + self.cash])
            self.total = total + self.cash
        logger.debug(f'== Total: {self.total}')

# ----------------------------------

def label_min_max(df, profit_threshold=0.08):
    # Columns for ideal case
    df['Trend'] = 1 # 0: downtrend, 1: uptrend - ML target
    df['Signal'] = -1 # 0-local max: sell, 1-local min: buy, -1: do nothing

    # Columns for benchmark
    df['Swing_Trend'] = 0 # 0: downtrend, 1: uptrend - Can get for sure
    df['Swing_Signal'] = -1 # 0: sell, 1: buy, -1: do nothing
    df['Swing_Assert'] = -1 # -1: not sure, 0/1: sure
    
    brick_size = df['Adj Close'].iat[0] * profit_threshold
    last_brick_price = df['Adj Close'].iat[0]
    trend = 0  # 0 for downtrend, 1 for uptrend    
    local_min = local_max = 0 # indices for tracing
    
    for i in range(1, len(df)):

        if trend == 1 and df['Adj Close'].iat[i] - last_brick_price >= brick_size: # uptrend one more brick
            local_min = local_max = i
            last_brick_price = df['Adj Close'].iat[i]            
        elif trend == 0 and last_brick_price - df['Adj Close'].iat[i] >= brick_size: # downtrend one more brick
            local_min = local_max = i
            last_brick_price = df['Adj Close'].iat[i]
        
        if trend == 1 and df['Adj Close'].iat[local_max] - df['Adj Close'].iat[i] >= brick_size: # End of uptrend 
            trend = 0
            df['Signal'].iat[local_max] = 0 
            for k in range(local_max, i + 1): df['Trend'].iat[k] = 0 # this dosn't happen to Swing
            last_brick_price = df['Adj Close'].iat[i]
            local_min = local_max = i    
            brick_size = df['Adj Close'].iat[local_max] * profit_threshold # new brick size        
        elif trend == 0 and df['Adj Close'].iat[i] - df['Adj Close'].iat[local_min] >= brick_size: # End of downtrend 
            trend = 1
            df['Signal'].iat[local_min] = 1 
            for k in range(local_min, i + 1): df['Trend'].iat[k] = 1 # this dosn't happen to Swing
            last_brick_price = df['Adj Close'].iat[i]
            local_min = local_max = i
            brick_size = df['Adj Close'].iat[local_min] * profit_threshold # new brick size
        
        df['Trend'].iat[i] = trend
        
        df['Swing_Trend'].iat[i] = trend
        if df['Swing_Trend'].iat[i] == 1 and df['Swing_Trend'].iat[i-1] == 0:
            df['Swing_Signal'].iat[i] = 1 # buy
        elif df['Swing_Trend'].iat[i] == 0 and df['Swing_Trend'].iat[i-1] == 1:
            df['Swing_Signal'].iat[i] = 0 # sell
            
        if df['Adj Close'].iat[i] >= df['Adj Close'].iat[local_max]:
            local_max = i
            if trend == 1: df['Swing_Assert'].iat[i] = 1
        if df['Adj Close'].iat[i] <= df['Adj Close'].iat[local_min]:
            local_min = i
            if trend == 0: df['Swing_Assert'].iat[i] = 0
    return df

# ----------------------------------

def n_day_regression(n, df, idxs):
    #variable
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan
    for idx in idxs:
        if idx > n:            
            y = df['Adj Close'][idx - n: idx].to_numpy()
            x = np.arange(0, n)
            #reshape
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            #calculate regression coefficient 
            lr = LinearRegression()
            lr.fit(x, y)  
            coef = lr.coef_[0][0]
            df[_varname_].iat[idx] = coef # JS: fix            
    return df

def get_llt(df, days=60):   
    alpha = 2 / (days + 1)
    prices = df['Adj Close']
    
    llt = pd.Series(index=prices.index, dtype='float64')
    # 需要至少两个价格点来计算LLT
    if len(prices) < 2:
        return prices
    # 初始化LLT的前两个值
    llt[0] = prices[0]
    llt[1] = prices[1]
    # 使用给定的公式计算接下来的LLT值
    for t in range(2, len(prices)):
        llt[t] = ((alpha - alpha**2 / 4) * prices[t] +
                  (alpha**2 / 2) * prices[t-1] -
                  (alpha - 3 * alpha**2 / 4) * prices[t-2] +
                  2 * (1 - alpha) * llt[t-1] -
                  (1 - alpha)**2 * llt[t-2])
    df['LLT'] = llt
    df['LLT_Ratio'] = (df['LLT'] - df['Adj Close']) / df['Adj Close']
    
    df['LLT_lag1'] = df['LLT'].shift(1)
    df['LLT_lag2'] = df['LLT'].shift(2)
    df['LLT_lag3'] = df['LLT'].shift(3)
    df['LLT_lag4'] = df['LLT'].shift(4)
    df['LLT_lag5'] = df['LLT'].shift(5)
    # LLT to lags ratio
    df['LLT_lag1_ratio'] = df['LLT'] / df['LLT_lag1']
    df['LLT_lag2_ratio'] = df['LLT'] / df['LLT_lag2']
    df['LLT_lag3_ratio'] = df['LLT'] / df['LLT_lag3'] 
    df['LLT_lag4_ratio'] = df['LLT'] / df['LLT_lag4'] 
    df['LLT_lag5_ratio'] = df['LLT'] / df['LLT_lag5']    
    return df.drop(columns=['LLT', 'LLT_lag1', 'LLT_lag2', 'LLT_lag3', 'LLT_lag4', 'LLT_lag5'])

def get_channels(df, mamode='sma', length=20, std=2.0, scalar=2):
    # 计算布林带（Bollinger Bands）
    length = 20  # 均线的周期
    std = 2.0  # 通道的宽度设定为几倍标准差
    mamode = 'sma'  # 均线的计算方式    
    channel = pd_ta.bbands(df['Adj Close'], length=length, std=std, mamode=mamode)
    columns = list(channel.columns)
    # BBL_20_2.0  BBM_20_2.0  BBU_20_2.0  BBB_20_2.0  BBP_20_2.0 = lower, mid, upper, bandwidth, percent
    for col in channel.columns: 
        df[col] = channel[col]
    for col in channel.columns[:3]: 
        df[f'{col}_Ratio'] = (df[f'{col}'] - df['Adj Close']) / df['Adj Close']
    
    # 计算肯特纳通道（Keltner Channels）
    length = 20  # 均线的周期
    scalar = 2  # 通道的宽度设定为几倍ATR
    mamode = 'ema'  # 均线的计算方式    
    channel = pd_ta.kc(df['High'], df['Low'], df['Adj Close'], length=length, scalar=scalar, mamode=mamode)
    columns += list(channel.columns)
    for col in channel.columns: 
        df[col] = channel[col]
        df[f'{col}_Ratio'] = (df[f'{col}'] - df['Adj Close']) / df['Adj Close']
    
    # 计算唐奇安通道（Donchian Channels）
    length = 20  # 最高价、最低价的周期
    channel = pd_ta.donchian(df['High'], df['Low'], lower_length=length, upper_length=length)
    # 计算霍尔特-温特通道（Holt-Winter Channel）
    channel = pd_ta.hwc(df['Adj Close'])
    columns += list(channel.columns)
    for col in channel.columns: 
        df[col] = channel[col]
        df[f'{col}_Ratio'] = (df[f'{col}'] - df['Adj Close']) / df['Adj Close']
    
    # 计算加速带（Acceleration Bands）
    length = 20  # 均线的周期
    mamode = 'sma'  # 均线的计算方式    
    channel = pd_ta.accbands(df['High'], df['Low'], df['Adj Close'], length=length, mamode=mamode)
    columns += list(channel.columns)
    for col in channel.columns: 
        df[col] = channel[col]
        df[f'{col}_Ratio'] = (df[f'{col}'] - df['Adj Close']) / df['Adj Close']
        
    return df.drop(columns=columns)

def get_rsrs(price_df, window_N=16, window_M=300, s=0.7):
    # 基础版的RSRS
    # 最高价和最低价的窗口长度
    window_N = 16
    
    # 初始化斜率和决定系数R-squared序列
    beta = np.full(price_df.shape[0], np.nan)
    r_squared = np.full(price_df.shape[0], np.nan)
    
    # 逐个滚动窗口计算
    for i in range(window_N-1, len(price_df)):
        # 获取窗口数据
        y = price_df['High'].iloc[i-window_N+1:i+1].values
        X = np.c_[np.ones(window_N), price_df['Low'].iloc[i-window_N+1:i+1].values]
    
        # 线性回归模型
        model = LinearRegression()
        model.fit(X, y)
    
        # 保存斜率和R-squared
        beta[i] = model.coef_[1]
        r_squared[i] = model.score(X, y)
    
    price_df['rsrs_beta'] = beta
    price_df['r_squared'] = r_squared
    
    # 标准分版的RSRS
    # 计算标准分的窗口长度
    window_M = 300
    # 计算滚动窗口的平均值和标准差
    rolling_mean = price_df['rsrs_beta'].rolling(window=window_M).mean()
    rolling_std = price_df['rsrs_beta'].rolling(window=window_M).std()
    
    # 计算斜率的Z-score值 = (当日斜率值 − 斜率均值) / 斜率标准差
    price_df['rsrs_zscore'] = (price_df['rsrs_beta'] - rolling_mean) / rolling_std
    
    # 修正标准分版的RSRS = 标准分RSRS * 决定系数
    price_df['rsrs_zscore_r2'] = price_df['rsrs_zscore'] * price_df['r_squared']
    
    # 右偏标准分版的RSRS = 修正标准分RSRS * 斜率
    price_df['rsrs_zscore_positive'] = price_df['rsrs_zscore_r2'] * price_df['rsrs_beta']
    
    # 根据RSRS择时
    rsrs_list = ['rsrs_zscore', 'rsrs_zscore_r2', 'rsrs_zscore_positive']
    rsrs_name = ['standard_RSRS', 'revised_RSRS', 'right_skew_RSRS']
    s = 0.7  # RSRS的阈值
    
    # 计算择时信号:RSRS值高于s时开仓，RSRS值低于-s时清仓，RSRS值在-s和s之间时维持先前的仓位
    for i in range(len(rsrs_list)):
        rsrs = rsrs_list[i]
        price_df[f'{rsrs_name[i]}_timing'] = (price_df[rsrs]>=s) * 1. + (price_df[rsrs]<=-s) * -1.

    return price_df.drop(columns=['rsrs_beta', 'r_squared'])

def add_indicators(df):   
    df['Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['Gain'] = [1 if r > 0 else -1 for r in df['Return']]
    df['PCT_Return'] = np.round((df['Adj Close'].pct_change()) * 100, 2)
    df['Normalized_Value'] = (df.Close - df.Low) / (df.High - df.Low + 10e-10)     

    # based on: https://www.kaggle.com/code/lusfernandotorres/trading-with-machine-learning-follow-the-trend
    # Features related to price behavior
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_AdjClose_Ratio'] = df['Open'] / df['Adj Close']
    df['Candle_to_Wick_Ratio'] = (df['Adj Close'] - df['Open']) / (df['High'] - df['Low'])
    upper_wick_size = df['High'] - df[['Open', 'Adj Close']].max(axis = 1)
    lower_wick_size = df[['Open', 'Adj Close']].min(axis = 1) - df['Low'] 
    df['Upper_to_Lower_Wick_Ratio'] = upper_wick_size /  lower_wick_size
       
    # Lagged values
    df['Close_lag1'] = df['Adj Close'].shift(1)
    df['Close_lag2'] = df['Adj Close'].shift(2)
    df['Close_lag3'] = df['Adj Close'].shift(3)
    df['Close_lag4'] = df['Adj Close'].shift(4)
    df['Close_lag5'] = df['Adj Close'].shift(5)
    
    # Close to lags ratio
    df['Close_lag1_ratio'] = df['Adj Close'] / df['Close_lag1']
    df['Close_lag2_ratio'] = df['Adj Close'] / df['Close_lag2']
    df['Close_lag3_ratio'] = df['Adj Close'] / df['Close_lag3'] 
    df['Close_lag4_ratio'] = df['Adj Close'] / df['Close_lag4'] 
    df['Close_lag5_ratio'] = df['Adj Close'] / df['Close_lag5']     
    df = df.drop(columns=['Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag4', 'Close_lag5'])
    
    #create regressions for 3, 5 and 10 days
    df = n_day_regression(3, df, list(range(len(df))))  
    df = n_day_regression(5, df, list(range(len(df))))  
    df = n_day_regression(10, df, list(range(len(df))))  
    df = n_day_regression(20, df, list(range(len(df))))  

    # based on: https://www.kaggle.com/code/lusfernandotorres/trading-with-machine-learning-classification
    # Adding Moving Averages
    sma_periods = [5, 10, 15, 20, 30, 50, 60, 80, 100]
    for period in sma_periods:
        sma = SMAIndicator(df['Adj Close'], window=period)
        df[f'SMA_{period}'] = sma.sma_indicator()
        
    # Adding Price to Moving Averages ratios    
    for period in sma_periods:
        df[f'SMA_{period}_ratio'] = df['Adj Close'] / df[f'SMA_{period}']
        df[f'SMA_{period}_prop'] = (df['Adj Close'] - df[f'SMA_{period}']) / df['Adj Close']

    for i in range(0, len(sma_periods) - 1):
        df[f'SMA_Delta_{sma_periods[i]}'] = (df[f'SMA_{sma_periods[i]}'] - df[f'SMA_{sma_periods[i+1]}']) / df['Adj Close']
                                             
    # Adding features derived from the indicators above    
    for ind1 in range(len(sma_periods) - 1):
        for ind2 in range(ind1 + 1, len(sma_periods)):
            df[f'SMA_{sma_periods[ind1]} vs SMA_{sma_periods[ind2]}'] = \
                (df[f'SMA_{sma_periods[ind1]}'] - df[f'SMA_{sma_periods[ind2]}']) / df[f'SMA_{sma_periods[ind2]}']
    
    for period in sma_periods:
        df = df.drop(columns=[f'SMA_{period}'])
        
    # Adding Exponential Moving Averages
    ema_periods = [5, 8, 12, 26, 50, 80, 100]
    for period in ema_periods:
        ema = EMAIndicator(df['Adj Close'], window=period)
        df[f'EMA_{period}'] = ema.ema_indicator()
        
    # Adding Price to Moving Averages ratios    
    for period in ema_periods:
        df[f'EMA_{period}_ratio'] = df['Adj Close'] / df[f'EMA_{period}']
        df[f'EMA_{period}_prop'] = (df['Adj Close'] - df[f'EMA_{period}']) / df['Adj Close']

    for i in range(0, len(ema_periods) - 1):
        df[f'EMA_Delta_{ema_periods[i]}'] = (df[f'EMA_{ema_periods[i]}'] - df[f'EMA_{ema_periods[i+1]}']) / df['Adj Close']        
        
    # Adding features derived from the indicators above    
    for ind1 in range(len(ema_periods) - 1):
        for ind2 in range(ind1 + 1, len(ema_periods)):
            df[f'EMA_{ema_periods[ind1]} vs EMA_{ema_periods[ind2]}'] = \
                (df[f'EMA_{ema_periods[ind1]}'] - df[f'EMA_{ema_periods[ind2]}']) / df[f'EMA_{ema_periods[ind2]}']
            
    for period in ema_periods:
        df = df.drop(columns=[f'EMA_{period}'])

    macd_object = MACD(close=df['Adj Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df['MACD'] = macd_object.macd()
    df['MACD_Signal'] = macd_object.macd_signal()
    df['MACD_Diff'] = macd_object.macd_diff()
    df['MACD_Diff_Ratio'] = macd_object.macd_diff() / macd_object.macd_signal()
    df['MACD Above MACD_Signal'] = (df['MACD'] - df['MACD_Signal']) / df['MACD_Signal']
    df['MACD_Diff Above Zero'] = (df['MACD_Diff'] > 0).astype(int)   
    df = df.drop(columns=['MACD', 'MACD_Signal', 'MACD_Diff'])
    
    # Adding RSI, CCI, and OBV
    df['RSI'] = RSIIndicator(df['Adj Close']).rsi() # window=14 by default
    df['RSI_Overbought_80'] = (df['RSI'] >= 80).astype(int)
    df['RSI_Overbought_70'] = (df['RSI'] >= 70).astype(int)
    df['RSI_Oversold_30'] = (df['RSI'] <= 30).astype(int)
    df['RSI_Oversold_20'] = (df['RSI'] <= 20).astype(int)
    
    df['CCI'] = CCIIndicator(df['High'], df['Low'], df['Close'], window=10, constant=0.015).cci() # window=20 by default
    df['CCI_High'] = (df['CCI'] >= 120).astype(int)
    df['CCI_Low'] = (df['CCI'] <= -120).astype(int)
    
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Adj Close'], volume=df['Volume']).on_balance_volume()
    df['OBV_Divergence_5_Days'] = df['OBV'].diff().rolling(10).sum() - df['Adj Close'].diff().rolling(5).sum()
    df['OBV_Divergence_10_Days'] = df['OBV'].diff().rolling(20).sum() - df['Adj Close'].diff().rolling(10).sum()
    
    # Add Bollinger Bands indicator
    bb = BollingerBands(df['Adj Close'])
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle_Ratio'] = (df['BB_Middle'] - df['Adj Close']) / df['Adj Close']
    df['BB_Upper_Ratio'] = (df['BB_Upper'] - df['Adj Close']) / df['Adj Close']
    df['BB_Lower_Ratio'] = (df['BB_Lower'] - df['Adj Close']) / df['Adj Close']
    df['Above_BB_Upper'] = (df['Adj Close'] >= df['BB_Upper']).astype(int)
    df['Below_BB_Lower'] = (df['Adj Close'] <= df['BB_Lower']).astype(int)
    df = df.drop(columns=['BB_Middle', 'BB_Upper', 'BB_Lower'])
    
    # Volatility features
    df['9_days_volatility'] = df['Adj Close'].pct_change().rolling(window=9).std()
    df['20_days_volatility'] = df['Adj Close'].pct_change().rolling(window=20).std()
    df['9_to_20_day_vol_ratio'] = df['9_days_volatility'] / df['20_days_volatility']
    df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Adj Close'], window=10).adx()
    df['ADI'] = AccDistIndexIndicator(df['High'], df['Low'], df['Adj Close'], df['Volume']).acc_dist_index()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Adj Close'], window=14).average_true_range()

    # LLT
    df = get_llt(df)    
    # Channels
    df = get_channels(df)
    # RSRS
    # df = get_rsrs(df)

    # Replacing infinite values by zeros
    df = df.replace([np.inf, -np.inf], 0)
    # Removing NaN values from the dataframe 
    df.dropna(inplace = True)
    return df

# ----------------------------------

def get_train_test_data_us(tickers=['qqq', 'spy'], start=None, end=None, profit_threshold=0.08):
    df = None
    count, total = 1, len(tickers)
    for stock in tickers:
        # logger.debug(f'Get Data Progress: {count} / {total} = {count/total:.2f}')
        try: 
            data = yf.download(stock, start=start, end=end, progress=False)
            data['Ticker'] = stock
            data = label_min_max(data, profit_threshold)
            data = add_indicators(data)              
            df = pd.concat([df, data], ignore_index=True) if df is not None else data          
        except Exception as ex:
            logger.debug(f'Exception in fetch_data for {stock}: {ex}')
        count += 1
    return df

def get_train_test_data_cn(tickers=['sh000001', '000300'], start=None, end=None, profit_threshold=0.08):
    df = None
    count, total = 1, len(tickers)
    for stock in tickers:
        logger.debug(f'Get Data Progress: {count} / {total} = {count/total:.2f}')
        try: 
            data = ak.stock_zh_index_daily(symbol=stock)
            data['Ticker'] = stock
            data = label_min_max(data, profit_threshold)
            data = add_indicators(data)              
            df = pd.concat([df, data], ignore_index=True) if df is not None else data    
        except Exception as ex:
            logger.debug(f'Exception in fetch_data for {stock}: {ex}')
        count += 1
    df['Date'] = pd.to_datetime(df['date'])
    if start is not None: 
        start = pd.to_datetime(start)
        df = df[(df['Date']>=start)]
    if end is not None: 
        end = pd.to_datetime(end)
        df = df[(df['Date']<=end)]
    df = df.sort_values('Date').set_index('Date')
    df = df.drop(columns=['date'])
    df.columns = [x.title() for x in df.columns]
    df['Adj Close'] = df['Close']
    cols = df.columns.tolist()
    cols[4], cols[5] = cols[5], cols[4] # switch Volume and Adj Close
    df = df[cols]
    return df

def get_train_test_data(market='us', *args, **kwargs):
    if market == 'us': return get_train_test_data_us(*args, **kwargs)
    elif market == 'cn': return get_train_test_data_cn(*args, **kwargs)

# ----------------------------------

def get_stock_current_price(stock):
    try:
        data = yf.download(stock, start=today, end=today+timedelta(days=1), progress=False)
        return data['Close'].values[-1]
    except Exception as ex:
        logger.debug(f'Exception when downloading data: {str(ex)}')
        return None

def check_sell(stock, hold, stop_loss, take_profit, threshold):
    curr_price = get_stock_current_price(stock)
    if curr_price is None: return 'hold', None
        
    # stop loss percent can be calculated based on gain/loss
    if curr_price < hold[3] * (1 - stop_loss): 
        logger.debug(f'{today} To sell for stop-loss: {stock}, at price {curr_price}, max is {hold[3]}')
        return 'sell', curr_price    

    # if curr_price > hold[1] * (1 + take_profit):
    #     logger.debug(f'{today} To sell for take-profit: {stock}, at price {curr_price}, buy price is {hold[1]}')
    #     return 'sell', curr_price    
    
    # use model to check for sell signal
    df = get_train_test_data('us', [stock], start=today-timedelta(days=150), end=today+timedelta(days=1))  
    if df is None or len(df) == 0: return 'hold', None
    columns = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'PCT_Return', 'Gain', 
               'Trend', 'Signal', 'Swing_Trend', 'Swing_Signal', 'Swing_Assert']
    df_selected = selector_sell.transform(df.drop(columns=columns))
    df_selected = df_selected[-2:, :]
    df_scaled = scaler_sell.transform(df_selected) 
    predictions = model_sell.predict(df_scaled)
    predictions_proba = model_sell.predict_proba(df_scaled)[:,1]
    predictions_proba_thresholded = 1 if predictions_proba[-1] > threshold else 0 # 1 for sell, 0 for hold
    logger.debug(f'{today} To sell for predict: {stock}, {threshold}, {predictions_proba}, {predictions_proba_thresholded}, {curr_price}')
    if predictions_proba_thresholded == 1: return 'sell', curr_price    
        
    if curr_price > hold[3]: hold[3] = curr_price
    return 'hold', curr_price  

def get_stock_pool():
    return triples_l # ['sqqq', 'yinn'] # fang_plus + etfs # triples #common_tickers # ['qqq', 'spy'] # 
    
def get_predict(stock, threshold):
    df = get_train_test_data('us', [stock], start=today-timedelta(days=150), end=today+timedelta(days=1))  
    if df is None or len(df) == 0: return None, None, None
    curr_price = df['Close'].values[-1]
    columns = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'PCT_Return', 'Gain', 
               'Trend', 'Signal', 'Swing_Trend', 'Swing_Signal', 'Swing_Assert']
    df_selected = selector_buy.transform(df.drop(columns=columns))
    df_selected = df_selected[-2:, :]
    df_scaled = scaler_buy.transform(df_selected) 
    # predictions = model.predict(df_scaled)
    predictions_proba = model_buy.predict_proba(df_scaled)[:,1]
    predictions_proba_thresholded = 1 if predictions_proba[-1] > threshold else 0 # 1 for buy, 0 for sell and hold
    # predictions_proba_thresholded = [1 if x > threshold 
    #                                  else 0 for x in predictions_proba] # 1 for buy, 0 for sell and hold    
    return predictions_proba[-1], predictions_proba_thresholded, curr_price
    
def get_stocks_to_buy(n, threshold):
    if n == 0: return []
    stocks_to_buy = []
    stock_pool = get_stock_pool()
    for stock in stock_pool:
        predictions_proba, prediction_thresholded, curr_price = get_predict(stock, threshold)
        logger.debug(f'{today} Predit on buy: {stock}, {threshold}, {predictions_proba}, {prediction_thresholded}, {curr_price}')
        if prediction_thresholded is None: continue
        if prediction_thresholded == 1: 
            stocks_to_buy.append((stock, curr_price, predictions_proba))
    stocks_to_buy.sort(key=lambda x:x[2], reverse=True)
    logger.debug(f'Stock to buy: {stocks_to_buy}')
    return stocks_to_buy[:n]

# ----------------------------------

def trading_job_sell():
    logger.debug('Go selling...')
    if today.isoweekday() in [6, 7]: return
    try:        
        # two options: don't sell if want to buy, or don't buy if want to sell
        # or: if hold in to buy but proba is lower than others, then sell it
        
        # check selling
        cash, holds = client.get_info() # {stock:[shares, buy_price, cost, max_price], }   
        if holds is not None and len(holds) > 0:  
            # way1: to buy then not sell
            stocks_to_buy = get_stocks_to_buy(max_stocks, min_threshold) # [(stock, curr_price, predictions_proba), ]
            stocks_to_check = list(set(holds) - set([stock[0] for stock in stocks_to_buy]))
            for stock in stocks_to_check:
                hold = holds.get(stock, None)
                if hold is not None:
                    action, price = check_sell(stock, hold, stop_loss, take_profit, max_threshold)
                    if action == 'sell': client.sell(stock, price)
                         
    except Exception as ex:
        logger.debug(f'{str(ex)}')


def trading_job_buy():
    logger.debug('Go buying...')
    if today.isoweekday() in [6, 7]: return
    try:                
        # check buying
        cash, holds = client.get_info() # {stock:[shares, buy_price, cost, max_price], }
        if cash < min_cash or len(holds) >= max_stocks: return
        cash -= min_cash
        # no need to get stocks to buy in this test, in practice need to do this because check buy and check sell happened at different time
        stocks_to_buy = get_stocks_to_buy(max_stocks, min_threshold) # [(stock, curr_price, predictions_proba), ]
      
        if len(stocks_to_buy) == 0: return
        num = min(len(stocks_to_buy), max_stocks - len(holds))
        count = num        
        for (stock, curr_price, predictions_proba) in stocks_to_buy[:num]:
            value = cash / count
            shares = value // curr_price
            if shares > 0: 
                client.buy(stock, curr_price, shares)
                logger.debug(f'{stock} is bought at {curr_price} with {shares} on {str(today)}')
                cash -= curr_price * shares
            count -= 1
                
    except Exception as ex:
        logger.debug(f'{str(ex)}')

def daily_log():  
    logger.debug('Go logging...')
    if today.isoweekday() in [6, 7]: return
    client.daily_update()

def weekly_log(): 
    logger.debug('Go weekly logging...')
    if today.isoweekday() in [1, 2, 3, 4, 5, 6]: return
    logger.debug('== History:')
    for hist in client.history: logger.debug(f'{hist}')
    logger.debug(f'== Final total: {client.total} ({(client.total - capital) * 100 / capital:.3f} % Gain)')
    log = [[x[0][:10], x[1] / capital] for x in client.log]
    log = pd.DataFrame(log, columns=['Date', 'Strategy']).set_index('Date')
    df = yf.download('tqqq', start=start_date, end=end_date+timedelta(days=1), progress=False)
    log = log.loc[[str(x)[:10] for x in df.index]]
    df['Benchmark'] = df['Adj Close'] / df['Adj Close'].iat[0]
    df['Strategy'] = [x for x in log['Strategy']]
    df = df[['Benchmark', 'Strategy']]
    df.plot(figsize=(10, 6), title='Performance')
    logger.debug(f"== Benchmark: {(df['Benchmark'].iat[-1] - 1) * 100: .3f} % Gain")

# https://stackoverflow.com/questions/10147455/how-to-send-an-email-with-gmail-as-provider-using-python
def send_email(subject, body):    
    gmail_user = 'john.sun.ca@gmail.com'
    gmail_app_password = 'ddub ojop flwo ythi'
    sent_from = gmail_user
    sent_to = ['john.sun.ca@gmail.com', ]
    sent_subject = subject
    sent_body = body    
    email_text = f'From: {sent_from}\nTo: {sent_to}\nSubject: {sent_subject}\n\n{sent_body}'
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_app_password)
        server.sendmail(sent_from, sent_to, email_text)
        server.close()   
        logger.debug(f'Email sent: {subject}')
    except Exception as ex:
        logger.debug(f'Email error: {str(ex)}')
# ----------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch_formatter = logging.Formatter('%(asctime)s - %(message)s')
# ch_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

# create file handler which logs event debug messages
fh = logging.FileHandler('trade.log')
fh.setLevel(logging.DEBUG)
# create formatters and add them to the handlers
fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
# fh_formatter = logging.Formatter('%(levelname)s - %(name)s - %(asctime)s - %(message)s', datefmt='%m-%d %H:%M')
fh.setFormatter(fh_formatter)
# add the handlers to logger
logger.addHandler(fh) 

def config_logger(filename):
    global logger, fh
    logger.removeHandler(fh)
    # create file handler which logs event debug messages
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    # create formatters and add them to the handlers
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    # fh_formatter = logging.Formatter('%(levelname)s - %(name)s - %(asctime)s - %(message)s', datefmt='%m-%d %H:%M')
    fh.setFormatter(fh_formatter)
    # add the handlers to logger
    logger.addHandler(fh)    

# ----------------------------------

triples_l = ['spxl', 'tqqq', 'tecl', 'soxl', 'nvdl', 'fngu']
# Load the selector, scaler, and classifier
mode = 'min'
selector_buy = joblib.load(f'{mode}_selector.joblib')
scaler_buy = joblib.load(f'{mode}_scaler.joblib')
model_buy = joblib.load(f'{mode}_classifier.joblib')
mode = 'max'
selector_sell = joblib.load(f'{mode}_selector.joblib')
scaler_sell = joblib.load(f'{mode}_scaler.joblib')
model_sell = joblib.load(f'{mode}_classifier.joblib')

tz = timezone('US/Eastern')
capital, start_date = 10000, datetime.now(tz)
min_cash, max_stocks, stop_loss, take_profit, min_threshold, max_threshold = 100, 3, 0.04, 100.0, 0.9, 0.9 
client = Client(capital)

# ----------------------------------

while True:    
    today = datetime.now(tz)
    current_time = today.strftime("%H:%M:%S")
    logger.debug(f'{today} - {current_time}')
    try:
        if '07:00:00' <= current_time < '08:00:00':             
            config_logger(filename=f'trade_{today.date()}.log')
        if today.isoweekday() in [1, 2, 3, 4, 5]:         
            if '10:00:00' <= current_time < '11:00:00': trading_job_sell()
            if '14:00:00' <= current_time < '15:00:00': trading_job_buy()
            if '22:00:00' <= current_time < '23:00:00': daily_log()
        if today.isoweekday() in [6]:
            if '14:00:00' <= current_time < '15:00:00':
                end_date = datetime.now(tz)+timedelta(days=1)
                weekly_log()
    except Exception as ex:
        logger.debug(str(ex))
    time.sleep(60*60)
