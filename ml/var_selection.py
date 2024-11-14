from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import os
import sys
from datetime import date, timedelta
import gc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import random
import requests
from math import sqrt
import math
import yaml
import traceback


# Define the expiration dates for ES Futures contracts
expiration_dates = {
    'ESH4': '2019-03-15',
    'ESM4': '2019-06-21',
    'ESU4': '2019-09-20',
    'ESZ4': '2019-12-20',
    'ESH5': '2020-03-20',
    'ESM5': '2020-06-19',
    'ESU5': '2020-09-18',
    'ESZ5': '2020-12-18',
    'ESH6': '2021-03-19',
    'ESM6': '2021-06-18',
    'ESU6': '2021-09-17',
    'ESZ6': '2021-12-17',
    'ESH7': '2022-03-18',
    'ESM7': '2022-06-17',
    'ESU7': '2022-09-16',
    'ESZ7': '2022-12-16',
    'ESH8': '2023-03-17',
    'ESM8': '2023-06-16',
    'ESU8': '2023-09-15',
    'ESZ8': '2023-12-15',
    'ESH9': '2024-03-15',
    'ESM9': '2024-06-21',
    'ESU9': '2024-09-20',
    'ESZ9': '2024-12-20'
}


# Function to determine the front month contract based on the current date
def get_front_month_contract(current_date):
    for contract, exp_date in expiration_dates.items():
        exp_date = datetime.strptime(exp_date, '%Y-%m-%d')
        if current_date <= exp_date - timedelta(days=10):
            return contract
    return 'ESZ4'  # Default to the last contract if no match found

def calculate_true_range(df):
    df['high-low'] = df['high'] - df['low']
    df['high-Prevclose'] = np.abs(df['high'] - df['close'].shift(1))
    df['low-Prevclose'] = np.abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['high-low', 'high-Prevclose', 'low-Prevclose']].max(axis=1)
    return df['TR']

def calculate_dm(df):
    df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                         np.maximum((df['high'] - df['high'].shift(1)), 0), 0)
    df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                         np.maximum((df['low'].shift(1) - df['low']), 0), 0)
    return df['+DM'], df['-DM']

def calculate_smoothed_values(df, period):
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df['+DM_Smoothed'] = df['+DM'].rolling(window=period).mean()
    df['-DM_Smoothed'] = df['-DM'].rolling(window=period).mean()
    return df['ATR'], df['+DM_Smoothed'], df['-DM_Smoothed']

def calculate_di(df):
    df['+DI'] = 100 * (df['+DM_Smoothed'] / df['ATR'])
    df['-DI'] = 100 * (df['-DM_Smoothed'] / df['ATR'])
    return df['+DI'], df['-DI']

def calculate_dx(df):
    df['DX'] = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    return df['DX']

def calculate_adx(df, period):
    df['ADX'] = df['DX'].rolling(window=period).mean()
    return df['ADX']

def calculate_adx_full(df, period=14):
    df = df.copy()
    df['TR'] = calculate_true_range(df)
    df['+DM'], df['-DM'] = calculate_dm(df)
    df['ATR'], df['+DM_Smoothed'], df['-DM_Smoothed'] = calculate_smoothed_values(df, period)
    df['+DI'], df['-DI'] = calculate_di(df)
    df['DX'] = calculate_dx(df)
    df['ADX'] = calculate_adx(df, period)
    return df[['TR', '+DM', '-DM', 'ATR', '+DI', '-DI', 'DX', 'ADX']]/100
class Database:
    def __init__(self,
                 start_trainning_date=None,
                 end_trainning_date=None,
                 start_trading_date=None,
                 end_trading_date=None,
                 minute_predict=30,
                 std_lookback=120,
                 lookback = 120,
                 k = 3,
                 train_split = 0.7,
                 start_trading=133000,  # 
                 end_trading=213000,
                 contract_trading = 'es',
                 cv = True,):

        self.start_trainning_date = start_trainning_date
        self.end_trainning_date = end_trainning_date
        self.start_trading_date = start_trading_date
        self.end_trading_date = end_trading_date
        
        self.start_trading = start_trading
        self.end_trading = end_trading

        self.minute_predict = minute_predict
        self.std_lookback = std_lookback
        self.lookback = lookback
        self.k = k
        self.train_split = train_split
        self.contract_trading = contract_trading
        self.cv = cv
        self.feature_required = []

    def _get_data(self,data_path,trading):
        merged_clean = pd.read_csv(data_path)
        merged_clean = merged_clean.ffill().copy()
        merged_clean['date_int'] = pd.to_datetime(merged_clean['ts_event']).dt.strftime('%Y%m%d').astype(int)
        merged_clean['time_int'] = pd.to_datetime(merged_clean['ts_event']).dt.strftime('%H%M%S').astype(int)
        # merged_clean['time_int'] = pd.to_datetime(merged_clean['ts_event']).dt.time.astype(int)

        if not trading:
            merged_clean = merged_clean[(merged_clean.date_int >= self.start_trainning_date)&(merged_clean.date_int <= self.end_trainning_date)].copy()
        else:
            merged_clean = merged_clean[(merged_clean.date_int >= self.start_trading_date)&(merged_clean.date_int <= self.end_trading_date)].copy()

        merged_clean.loc[:,f'period_return'] = np.log(merged_clean.loc[:,'close']).diff()

        #trainning data
        merged_clean.loc[:,f'period_normalized'] = merged_clean.loc[:,f'period_return'] / merged_clean.loc[:,f'period_return'].rolling(self.std_lookback).std()

        merged_clean.loc[:,f'period_return_shift'] = np.log(merged_clean.loc[:,'close'].shift(-self.minute_predict)) - np.log(merged_clean.loc[:,'close'])

        merged_clean.loc[:,f'period_return_shift'] = merged_clean.loc[:,f'period_return_shift'] / merged_clean.loc[:,f'period_return'].rolling(self.std_lookback).std()

        merged_clean.loc[:,f'period_forward'] = merged_clean.loc[:,'close'].diff()

        merged_clean.loc[:,'trading_time'] = merged_clean['time_int'].copy()

        #std
        merged_clean.loc[:,f'std_10'] = merged_clean.loc[:,f'period_return'].ewm(span = 10,min_periods=10).std()
        merged_clean.loc[:,f'std_30'] = merged_clean.loc[:,f'period_return'].ewm(span = 30,min_periods=30).std()
        merged_clean.loc[:,f'std_60'] = merged_clean.loc[:,f'period_return'].ewm(span = 60,min_periods=60).std()
        merged_clean.loc[:,f'std_120'] = merged_clean.loc[:,f'period_return'].ewm(span = 120,min_periods=120).std()

        #backward return
        merged_clean.loc[:,f'10_norm'] = (merged_clean.loc[:,'close']/(merged_clean.loc[:,'close'].shift(10))-1)/merged_clean.loc[:,f'std_10']/np.sqrt(10)
        merged_clean.loc[:,f'30_norm'] = (merged_clean.loc[:,'close']/(merged_clean.loc[:,'close'].shift(30))-1)/merged_clean.loc[:,f'std_30']/np.sqrt(30)
        merged_clean.loc[:,f'60_norm'] = (merged_clean.loc[:,'close']/(merged_clean.loc[:,'close'].shift(60))-1)/merged_clean.loc[:,f'std_60']/np.sqrt(60)
        merged_clean.loc[:,f'120_norm'] = (merged_clean.loc[:,'close']/(merged_clean.loc[:,'close'].shift(120))-1)/merged_clean.loc[:,f'std_120']/np.sqrt(120)

        #macd
        merged_clean.loc[:,f'10_macd'] = merged_clean.loc[:,'close'].ewm(halflife=10).mean()-merged_clean.loc[:,'close'].ewm(halflife=30).mean()
        merged_clean.loc[:,f'10_q'] = merged_clean.loc[:,f'10_macd'] / merged_clean.loc[:,'close'].rolling(30).std()
        merged_clean.loc[:,f'10_y'] = merged_clean.loc[:,f'10_q'] / merged_clean.loc[:,'10_q'].rolling(120).std()

        merged_clean.loc[:,f'20_macd'] = merged_clean.loc[:,'close'].ewm(halflife=20).mean()-merged_clean.loc[:,'close'].ewm(halflife=60).mean()
        merged_clean.loc[:,f'20_q'] = merged_clean.loc[:,f'20_macd'] / merged_clean.loc[:,'close'].rolling(60).std()
        merged_clean.loc[:,f'20_y'] = merged_clean.loc[:,f'20_q'] / merged_clean.loc[:,'20_q'].rolling(240).std()

        merged_clean.loc[:,f'30_macd'] = merged_clean.loc[:,'close'].ewm(halflife=30).mean()-merged_clean.loc[:,'close'].ewm(halflife=90).mean()
        merged_clean.loc[:,f'30_q'] = merged_clean.loc[:,f'30_macd'] / merged_clean.loc[:,'close'].rolling(90).std()
        merged_clean.loc[:,f'30_y'] = merged_clean.loc[:,f'30_q'] / merged_clean.loc[:,'30_q'].rolling(360).std()

        merged_clean.loc[:,f'60_macd'] = merged_clean.loc[:,'close'].ewm(halflife=60).mean()-merged_clean.loc[:,'close'].ewm(halflife=180).mean()
        merged_clean.loc[:,f'60_q'] = merged_clean.loc[:,f'60_macd'] / merged_clean.loc[:,'close'].rolling(180).std()
        merged_clean.loc[:,f'60_y'] = merged_clean.loc[:,f'60_q'] / merged_clean.loc[:,'60_q'].rolling(360).std()

        #vwap
        merged_clean['volume_30'] = merged_clean['volume'].rolling(30).sum()
        merged_clean['pv_30'] = (merged_clean['close'] * merged_clean['volume']).rolling(30).sum()
        merged_clean['VWAP_30'] = merged_clean['pv_30'] / merged_clean['volume_30']
        merged_clean['vwap_30_diff'] = np.log(merged_clean['VWAP_30']) - np.log(merged_clean['close']) 

        merged_clean['volume_60'] = merged_clean['volume'].rolling(60).sum()
        merged_clean['pv_60'] = (merged_clean['close'] * merged_clean['volume']).rolling(60).sum()
        merged_clean['VWAP_60'] = merged_clean['pv_60'] / merged_clean['volume_60']
        merged_clean['vwap_60_diff'] = np.log(merged_clean['VWAP_60']) - np.log(merged_clean['close']) 

        merged_clean['volume_90'] = merged_clean['volume'].rolling(90).sum()
        merged_clean['pv_90'] = (merged_clean['close'] * merged_clean['volume']).rolling(90).sum()
        merged_clean['VWAP_90'] = merged_clean['pv_90'] / merged_clean['volume_90']
        merged_clean['vwap_90_diff'] = np.log(merged_clean['VWAP_90']) - np.log(merged_clean['close']) 

        #daily vwap
        merged_clean['volume_cumsum'] = merged_clean.groupby('date_int').volume.cumsum()
        merged_clean['pv'] = (merged_clean['close'] * merged_clean['volume'])
        merged_clean['pv_cumsum'] = merged_clean.groupby('date_int').pv.cumsum()

        merged_clean['vwap'] = merged_clean['pv_cumsum'] / merged_clean['volume_cumsum']
        merged_clean['vwap_diff'] = np.log(merged_clean['vwap']) - np.log(merged_clean['close']) 

        #vwap diff norm
        merged_clean['vwap_30_diff_norm'] = merged_clean['vwap_30_diff'] / merged_clean['vwap_30_diff'].rolling(120).std()
        merged_clean['vwap_60_diff_norm'] = merged_clean['vwap_60_diff'] / merged_clean['vwap_60_diff'].rolling(120).std()
        merged_clean['vwap_90_diff_norm'] = merged_clean['vwap_90_diff'] / merged_clean['vwap_90_diff'].rolling(120).std()
        merged_clean['vwap_diff_norm'] = merged_clean['vwap_diff'] / merged_clean['vwap_diff'].rolling(120).std()

        #volumne price trend
        merged_clean['price_change_pct'] = merged_clean['close'].pct_change()
        merged_clean['vpt'] = merged_clean['price_change_pct'] * merged_clean['volume']

        merged_clean['vpt_10'] = merged_clean['vpt'].ewm(span = 10,min_periods=10).mean()
        merged_clean['vpt_30'] = merged_clean['vpt'].ewm(span = 30,min_periods=30).mean()
        merged_clean['vpt_90'] = merged_clean['vpt'].ewm(span = 90,min_periods=90).mean()

        #calculate adx
        merged_clean[['TR_30', '+DM_30', '-DM_30', 'ATR_30', '+DI_30', '-DI_30', 'DX_30', 'ADX_30']] = calculate_adx_full(merged_clean, period=30)
        merged_clean[['TR_60', '+DM_60', '-DM_60', 'ATR_60', '+DI_60', '-DI_60', 'DX_60', 'ADX_60']] = calculate_adx_full(merged_clean, period=60)
        merged_clean[['TR_90', '+DM_90', '-DM_90', 'ATR_90', '+DI_90', '-DI_90', 'DX_90', 'ADX_90']] = calculate_adx_full(merged_clean, period=90)

        #absolute return and volumne correlation
        merged_clean['vol_mean'] = merged_clean.groupby('trading_time').volume.rolling(15).mean().reset_index(level=0, drop=True)
        merged_clean['vol_std'] = merged_clean.groupby('trading_time').volume.rolling(15).std().reset_index(level=0, drop=True)
        merged_clean['vol_norm'] = (merged_clean.volume - merged_clean['vol_mean'])/merged_clean['vol_std']


        merged_clean['re_vol_corr_30'] = merged_clean['period_return'].abs().rolling(window=30, min_periods=30).corr(merged_clean['vol_norm'])
        merged_clean['re_vol_corr_60'] = merged_clean['period_return'].abs().rolling(window=60, min_periods=60).corr(merged_clean['vol_norm'])
        merged_clean['re_vol_corr_90'] = merged_clean['period_return'].abs().rolling(window=90, min_periods=90).corr(merged_clean['vol_norm'])

        #volume self correlation
        merged_clean['vol_diff_t_1'] = merged_clean['volume'] - merged_clean['volume'].shift(1)
        merged_clean['vol_diff_t'] = merged_clean['volume'].shift(1) - merged_clean['volume'].shift(2)

        merged_clean['vol_diff_t_1_pos'] = np.where(merged_clean['vol_diff_t_1']>0,merged_clean['vol_diff_t_1'],np.nan)
        merged_clean['vol_diff_t_pos'] = np.where(merged_clean['vol_diff_t']>0,merged_clean['vol_diff_t'],np.nan)


        merged_clean['rolling_corr'] = merged_clean['vol_diff_t_1_pos'].rolling(window=120,min_periods=1).corr(merged_clean['vol_diff_t_pos'])

        merged_clean['rolling_corr_mean'] = merged_clean.groupby('trading_time')['rolling_corr'].rolling(15).mean().reset_index(level=0, drop=True)
        merged_clean['rolling_corr_std'] = merged_clean.groupby('trading_time')['rolling_corr'].rolling(15).std().reset_index(level=0, drop=True)
        merged_clean['rolling_corr_norm'] = (merged_clean.rolling_corr - merged_clean['rolling_corr_mean'])/merged_clean['rolling_corr_std']

        merged_clean['vol_diff_t_1_neg'] = np.where(merged_clean['vol_diff_t_1']<0,merged_clean['vol_diff_t_1'],np.nan)
        merged_clean['vol_diff_t_neg'] = np.where(merged_clean['vol_diff_t']<0,merged_clean['vol_diff_t'],np.nan)


        merged_clean['rolling_corr_neg'] = merged_clean['vol_diff_t_1_neg'].rolling(window=120,min_periods=1).corr(merged_clean['vol_diff_t_neg'])

        merged_clean['rolling_corr_neg_mean'] = merged_clean.groupby('trading_time')['rolling_corr_neg'].rolling(15).mean().reset_index(level=0, drop=True)
        merged_clean['rolling_corr_neg_std'] = merged_clean.groupby('trading_time')['rolling_corr_neg'].rolling(15).std().reset_index(level=0, drop=True)
        merged_clean['rolling_corr_neg_norm'] = (merged_clean.rolling_corr_neg - merged_clean['rolling_corr_neg_mean'])/merged_clean['rolling_corr_neg_std']

        merged_clean['CDVDV'] = merged_clean['rolling_corr_neg_norm'] + merged_clean['rolling_corr_norm'] 


        #intraday move
        merged_clean['daily_high'] = merged_clean.groupby('date_int').high.cummax()
        print(merged_clean['daily_high'])
        merged_clean['daily_low'] = merged_clean.groupby('date_int').low.cummin()

        daily_close = merged_clean.groupby('date_int').close.last().shift(1).rename('daily_close')

        merged_clean = merged_clean.merge(daily_close,on='date_int',how='left')


        merged_clean['intraday_move'] = (merged_clean['daily_high'] - merged_clean['daily_low']) / merged_clean['daily_close']

        #smart money vwap ratio
        merged_clean['smart_money'] = merged_clean['period_return'].abs() / (merged_clean['volume'] ** 0.25)

        merged_clean['smart_money_top_300'] = merged_clean['smart_money'].rolling(window=300).quantile(0.8)
        merged_clean['smart_money_pv_300'] = np.where(merged_clean['smart_money'] > merged_clean['smart_money_top_300'],
                                                        merged_clean['volume']*merged_clean['close'],0)
        merged_clean['smart_money_v_300'] = np.where(merged_clean['smart_money'] > merged_clean['smart_money_top_300'],
                                                        merged_clean['volume'],0)
        merged_clean['smart_money_vwap_300'] = merged_clean['smart_money_pv_300'].rolling(300).sum() / merged_clean['smart_money_v_300'].rolling(300).sum()
        merged_clean['vwap_300'] = (merged_clean['volume']*merged_clean['close']).rolling(300).sum() / (merged_clean['volume']).rolling(300).sum()
        merged_clean['sm_vwap_ratio_300'] = merged_clean['smart_money_vwap_300'] / merged_clean['vwap_300']


        merged_clean['smart_money_top_200'] = merged_clean['smart_money'].rolling(window=200).quantile(0.8)
        merged_clean['smart_money_pv_200'] = np.where(merged_clean['smart_money'] > merged_clean['smart_money_top_200'],
                                                        merged_clean['volume']*merged_clean['close'],0)
        merged_clean['smart_money_v_200'] = np.where(merged_clean['smart_money'] > merged_clean['smart_money_top_200'],
                                                        merged_clean['volume'],0)
        merged_clean['smart_money_vwap_200'] = merged_clean['smart_money_pv_200'].rolling(200).sum() / merged_clean['smart_money_v_300'].rolling(200).sum()
        merged_clean['vwap_200'] = (merged_clean['volume']*merged_clean['close']).rolling(200).sum() / (merged_clean['volume']).rolling(200).sum()
        merged_clean['sm_vwap_ratio_200'] = merged_clean['smart_money_vwap_200'] / merged_clean['vwap_200']


        merged_clean['smart_money_top_100'] = merged_clean['smart_money'].rolling(window=100).quantile(0.8)
        merged_clean['smart_money_pv_100'] = np.where(merged_clean['smart_money'] > merged_clean['smart_money_top_100'],
                                                        merged_clean['volume']*merged_clean['close'],0)
        merged_clean['smart_money_v_100'] = np.where(merged_clean['smart_money'] > merged_clean['smart_money_top_100'],
                                                        merged_clean['volume'],0)
        merged_clean['smart_money_vwap_100'] = merged_clean['smart_money_pv_100'].rolling(100).sum() / merged_clean['smart_money_v_100'].rolling(100).sum()
        merged_clean['vwap_100'] = (merged_clean['volume']*merged_clean['close']).rolling(100).sum() / (merged_clean['volume']).rolling(100).sum()
        merged_clean['sm_vwap_ratio_100'] = merged_clean['smart_money_vwap_100'] / merged_clean['vwap_100']

        #vol ration askew
        merged_clean['10_daily_vol_sum'] = merged_clean.groupby('date_int').volume.rolling(10).sum().reset_index(level=0, drop=True)
        merged_clean['vol_ratio'] = merged_clean['volume'] / merged_clean['10_daily_vol_sum']

        merged_clean['vol_ratio_100_skew'] = merged_clean['vol_ratio'].rolling(100).skew()
        merged_clean['vol_ratio_200_skew'] = merged_clean['vol_ratio'].rolling(200).skew()
        merged_clean['vol_ratio_300_skew'] = merged_clean['vol_ratio'].rolling(300).skew()
        print(np.sum(merged_clean.loc[:,f'period_return']))

        feature_required = ['period_return_shift','period_forward','period_normalized','std_10','std_30','std_60','std_120',
                                      '10_norm','30_norm','60_norm','120_norm',
                                      '10_y','20_y','30_y','60_y',
                                      'vwap_30_diff_norm','vwap_60_diff_norm','vwap_90_diff_norm','vwap_diff_norm',
                                      'vpt_10','vpt_30','vpt_90',
                                      'TR_30', '+DM_30', '-DM_30', 'ATR_30', '+DI_30', '-DI_30', 'DX_30', 'ADX_30',
                                      'TR_60', '+DM_60', '-DM_60', 'ATR_60', '+DI_60', '-DI_60', 'DX_60', 'ADX_60',
                                      'TR_90', '+DM_90', '-DM_90', 'ATR_90', '+DI_90', '-DI_90', 'DX_90', 'ADX_90',
                                      're_vol_corr_30','re_vol_corr_60','re_vol_corr_90',
                                      'rolling_corr_norm','rolling_corr_neg_norm','CDVDV',
                                      'intraday_move','sm_vwap_ratio_300','sm_vwap_ratio_200','sm_vwap_ratio_100',
                                      'vol_ratio_100_skew','vol_ratio_200_skew','vol_ratio_300_skew']

        merged_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged_clean = merged_clean.dropna(subset=feature_required)

        #use all the data to train, maybe will improve performance
        # merged_clean = merged_clean[(merged_clean.time_int>=self.start_trading)&(merged_clean.time_int<=self.end_trading) ].copy()
    
        merged_clean.loc[:,'time_int'] = merged_clean['time_int']+ merged_clean['date_int']*1000000

        merged_clean = merged_clean.reset_index(drop=True)

        return merged_clean


    
    def create_dataloader(self,batch_size = 128,shuffle=False):
        try:
            current_dir = os.path.dirname(__file__)
            data_path = os.path.join(current_dir, 'data', f'{self.contract_trading}_minutebar_data.csv')
            training_df = self._get_data(data_path,trading=False)
        except FileNotFoundError:
            print(f'{self.contract_trading}_minutebar_data.csv not found')
            return None,None,None

        night_index = list(training_df[(training_df.trading_time<self.start_trading)|(training_df.trading_time>self.end_trading) ].index)

        filtered_index = list(training_df[~training_df.index.isin(night_index)].index)

        if self.cv:
            samples_in_fold = len(filtered_index)// self.k
            split = int(samples_in_fold*self.train_split)

            trainning_index_all = []
            validation_index_all = []

            for i in range(self.k):
                trainning_index = filtered_index[i*samples_in_fold:i*samples_in_fold+split]
                validation_index = filtered_index[i*samples_in_fold+split:(i+1)*samples_in_fold]

                trainning_index_all = trainning_index_all + trainning_index
                validation_index_all = validation_index_all + validation_index
        else:
            split = int(len(filtered_index)*self.train_split)

            trainning_index_all = filtered_index[:split]
            validation_index_all = filtered_index[split:]

        print('training_df')
        print(training_df)

        trainning_dataset = BTDataset(training_df,trainning_index_all,self.lookback,self.minute_predict)
        trainning_dataloader = DataLoader(trainning_dataset, batch_size=batch_size, shuffle=shuffle)
        
        #get validation dataloader
        validation_dataset = BTDataset(training_df,validation_index_all,self.lookback,self.minute_predict)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)

        #get trading dataloader
        try:
            current_dir = os.path.dirname(__file__)
            data_path = os.path.join(current_dir, 'data', f'{self.contract_trading}_minutebar_data.csv')
            trading_df = self._get_data(data_path,trading=True)
        except FileNotFoundError:
            print(f'{self.contract_trading}_minutebar_data.csv not found')
        trading_index_all = list(trading_df[(trading_df.trading_time>=self.start_trading)&(trading_df.trading_time<=self.end_trading) ].index)
        
        trading_dataset = BTDataset(trading_df,trading_index_all,self.lookback,self.minute_predict)
        trading_dataloader = DataLoader(trading_dataset, batch_size=batch_size, shuffle=False)

        return trainning_dataloader,validation_dataloader,trading_dataloader

class BTDataset(Dataset):
    def __init__(self,all_dfs,index_list,lookback,minute_predict):
        super().__init__()
        self.all_dfs = all_dfs
        self.index_list = index_list
        self.lookback = lookback
        self.minute_predict = minute_predict

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):

        sample_index = self.index_list[index]
        lookback_df = self.all_dfs.loc[sample_index-self.lookback+1:sample_index]
        #lookback_df = self.all_dfs.loc[:sample_index].iloc[-self.lookback:]

        # if len(lookback_df) < self.lookback:
        #     print(self.all_dfs.loc[:sample_index])
        #     print(lookback_df)
        #     raise ValueError

        if len(lookback_df) < self.lookback:
            padding = pd.DataFrame(0, index=np.arange(self.lookback-len(lookback_df)), columns=lookback_df.columns)
            lookback_df = pd.concat([padding,lookback_df], ignore_index=True)

        raw_data = lookback_df.loc[:,['period_normalized','std_10','std_30','std_60','std_120',
                                      '10_norm','30_norm','60_norm','120_norm',
                                      '10_y','20_y','30_y','60_y',
                                      'vwap_30_diff_norm','vwap_60_diff_norm','vwap_90_diff_norm','vwap_diff_norm',
                                      'vpt_10','vpt_30','vpt_90',
                                      'TR_30', '+DM_30', '-DM_30', 'ATR_30', '+DI_30', '-DI_30', 'DX_30', 'ADX_30',
                                      'TR_60', '+DM_60', '-DM_60', 'ATR_60', '+DI_60', '-DI_60', 'DX_60', 'ADX_60',
                                      'TR_90', '+DM_90', '-DM_90', 'ATR_90', '+DI_90', '-DI_90', 'DX_90', 'ADX_90',
                                      're_vol_corr_30','re_vol_corr_60','re_vol_corr_90',
                                      'rolling_corr_norm','rolling_corr_neg_norm','CDVDV',
                                       'intraday_move','sm_vwap_ratio_300','sm_vwap_ratio_200','sm_vwap_ratio_100',
                                      'vol_ratio_100_skew','vol_ratio_200_skew','vol_ratio_300_skew']]
        
        

        predict_df = self.all_dfs.loc[sample_index:sample_index]


        return {'raw_data':torch.from_numpy(raw_data.values).float(),
                'label':torch.from_numpy(predict_df['period_return_shift'].values).float(),
                'real_return':torch.from_numpy(predict_df['period_forward'].values).float(),
                'timeint':torch.from_numpy(lookback_df['trading_time'].values).float(),
                'dateint':torch.from_numpy(lookback_df['date_int'].values).float(),}
class GLU(torch.nn.Module):
    def __init__(self, dim_input,d_model):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(dim_input, d_model)
        self.fc2 = nn.Linear(dim_input, d_model)
    
    def forward(self, x):
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)
    
class Variable_selection(nn.Module):
    def __init__(self,d_model, window_size,dropout=0.1):
        super(Variable_selection,self).__init__()

        self.gate = GLU(window_size,d_model)

        self.dropout = nn.Dropout(dropout)

        self.weight_ffc = nn.Linear(d_model,1)
        # self.Sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        #x(B,T,N)
        x = x.permute(0,2,1) #(B,N,T)

        x = self.gate(x)#(B,N,D)
        x = self.dropout(x)

        weight = self.weight_ffc(x).squeeze(-1)#(B,N)

        # weight = self.Sigmoid(weight)
        weight = self.softmax(weight)
        return weight
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
    
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)
    

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

def add_weight_decay(
    model,
    weight_decay=1e-5,
    skip_list=("bias", "bn", "LayerNorm.bias", "LayerNorm.weight"),
):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
class IT(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, 
                 seq_len,
                 pred_len,
                 d_model,
                 n_heads,
                 dropout=0.1,
                 d_ff=2048,
                 activation = 'gelu',
                 e_layers = 2):
        super(IT, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed_type='fixed', freq='h',
                                                    dropout=dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc =  x_enc - means

        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates 
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class IM(nn.Module):
    def __init__(self, ts_dim, num_features,num_filters = 64, bias = False):
        super(IM,self).__init__()
        
        self.ts_dim = ts_dim
        self.num_features = num_features
        self.act_fcn = nn.ELU()

        self.cnn_layers1 = nn.Sequential(nn.Conv2d(1,num_filters,(1,1),padding = 'same', bias= bias),
                                        self.act_fcn,
                                        nn.Conv2d(num_filters, num_filters, (ts_dim,1),padding = 'same', bias= bias, groups = num_filters),
                                        self.act_fcn,
                                        nn.LayerNorm(num_features),
                                        )
        self.cnn_layers2 = nn.Sequential(nn.Conv2d(1,num_filters,(1,1),padding = 'same', bias= bias),
                                        self.act_fcn,
                                        nn.Conv2d(num_filters, num_filters, (1,num_features),padding = 'same', bias= bias, groups = num_filters),
                                        self.act_fcn,
                                        nn.LayerNorm(num_features),
                                        )
        self.cnn_layers3 = nn.Sequential(nn.Conv2d(1,num_filters,(1,1),padding = 'same', bias= bias),
                                        self.act_fcn,
                                        nn.Conv2d(num_filters, num_filters, (ts_dim,num_features),padding = 'same', bias= bias, groups = num_filters),
                                        self.act_fcn,
                                        nn.LayerNorm(num_features),
                                        )
        self.cnn_layers4 = nn.Sequential(nn.Conv2d(1,num_filters,(1,1),padding = 'same', bias= bias),
                                        self.act_fcn,
                                        nn.LayerNorm(num_features),
                                        )
        
        self.cnn_dimreduc = nn.Sequential(nn.Conv2d(num_filters*4,1, (1,1),bias= bias),
                                        self.act_fcn,
                                         )

    def forward(self,x):
        x = x.unsqueeze(1)

        x_input1 = self.cnn_layers1(x)
        x_input2 = self.cnn_layers2(x)
        x_input3 = self.cnn_layers3(x)
        x_input4 = self.cnn_layers4(x)
        
        x = torch.cat((x_input1,
                       x_input2,
                       x_input3,
                       x_input4)
                      ,dim = 1)
        
        x = self.cnn_dimreduc(x)


        x = x.squeeze(1)

        return x
    
class ITEncoder(nn.Module):
    def __init__(self,d_model,nhead,layers, window_size,num_features,ts_dim,num_filters,dropout=0.1, bias=False):
        super(ITEncoder,self).__init__()

        self.it = IT(seq_len=window_size,pred_len=1,d_model=d_model,n_heads=nhead,dropout=dropout,e_layers=layers,d_ff=d_model*2)
        self.im = IM(ts_dim=ts_dim,num_features=num_features,num_filters=num_filters)

        self.vs = Variable_selection(d_model,window_size,dropout)

        self.ffc = nn.Sequential(nn.Linear(num_features,1),
                                 nn.Tanh())

    def forward(self,batch):
        input = batch['raw_data']

        weight = self.vs(input).unsqueeze(1)
        weighted_input = input*weight

        x = self.im(weighted_input)

        x = self.it(x,None,None,None)
        x = x.squeeze(1)
        x = self.ffc(x)
        return {'trades':x}
    
class Trainner_V2:
    def __init__(self,
                 model,
                 optim,
                 scaler,
                 device,
                 lr,
                 loss_fn,
                 config,
                 trainning_dataloader,
                 validation_dataloader,
                 trading_dataloader,
                 scheduler_name = None,
                 scheduler_max_step = 20,
                 scheduler_min_lr = 1e-9,
                 max_epoch = 50,
                 exit_count = 15,
                 model_name = 'best_model_weights.pt',
                 output_filaname = 'output.txt',
                 backtest_name = 'backtest.png',
                 show = True,
                 gradient_clip = True,
                 max_gradient_norm = 1,
                 gamma = 2,
                 poly_power = 2,
                 mixed_percision_type = torch.bfloat16
                 ):
        '''
        compare to first version of trainner, this one use input and output and loss all use dicts so it can be generalized
        '''

        self.model = model.to(device)
        self.config = config
        self.trainning_dataloader = trainning_dataloader
        self.validation_dataloader = validation_dataloader
        self.trading_dataloader = trading_dataloader
        self.device = device
        self.optim = optim
        self.loss_fn = loss_fn
        self.lr = lr
        self.scheduler_name = scheduler_name
        self.max_epcoh = max_epoch
        self.exit_count = exit_count
        self.validation_loss_list = []
        self.trainning_loss_list = []
        self.model_name = model_name
        self.output_filaname = output_filaname
        self.show = show
        self.gradient_clip = gradient_clip
        self.max_gradient_norm = max_gradient_norm
        self.scaler = scaler
        self.mixed_percision_type = mixed_percision_type
        self.backtest_name = backtest_name

        if self.scheduler_name == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=scheduler_max_step, eta_min=scheduler_min_lr)

        elif self.scheduler_name == 'expo':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=gamma)

        elif self.scheduler_name == 'poly':
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optim, power=poly_power,total_iters=max_epoch)


    def log_print(self,out):
        if self.show:
            with open(self.output_filaname, "a") as f:
                print(out, file=f)
                print(out)

    def validation(self):

        loss_list = []
        element_loss = []

        sharpe_list = []
        validation_return = []

        final_df = []

        with torch.no_grad():
            self.model.eval()
            if self.show:
                pbar = tqdm(self.validation_dataloader, desc='Validation',unit='batch')
            else:
                pbar = self.validation_dataloader

            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                print(batch)
                label = batch['label']

                output_dict = self.model(batch)


                loss_dict = self.loss_fn(output_dict,label)

                loss = loss_dict['loss']
                loss_list.append(loss.cpu().numpy())

                element_loss.append([i.item() for i in loss_dict.values()])

                trades = output_dict['trades']
                batch_return = (trades*label).mean(dim=1)

                validation_return.append(batch_return.cpu().detach().numpy().reshape(-1))

                #start trading the validation dataloader
                timeint = batch['timeint']

                timeint_array = timeint[:,-1].cpu().numpy().reshape(-1)
                dateint = batch['dateint'][:,-1].cpu().numpy().reshape(-1)

                real_return = batch['real_return'].cpu().numpy().reshape(-1)

                trading_df = pd.DataFrame({'trade':trades.cpu().numpy().reshape(-1),
                                                    'real_return':real_return,
                                                    'dateint':dateint,
                                                    'trading_time':timeint_array,})
                # print(trading_df)
                # raise ValueError
                final_df.append(trading_df)

        #backtest on valid
        final_df = pd.concat(final_df).reset_index(drop=True)

        final_df_copy = final_df.copy()

        final_df_copy = final_df_copy[(final_df_copy.trading_time < 170000)&(final_df_copy.trading_time > 90000)].reset_index(drop=True).copy()

        final_df_copy.loc[:,'vol_weight'] =  0.15 / (final_df_copy.real_return.rolling(self.config['std_lookback']).std() * np.sqrt(252*24*60))

        final_df_copy.loc[:,'raw_trade'] = final_df_copy.trade.rolling(self.config['lookback_period']).sum()

        final_df_copy.loc[:,'weight'] = final_df_copy.loc[:,'raw_trade'] * final_df_copy.loc[:,'vol_weight']

        final_df_copy.loc[:,'pnl'] = final_df_copy.loc[:,'weight']  * final_df_copy['real_return'].shift(-1)

        daily_df = final_df_copy.groupby('dateint').pnl.sum()

        sharpe = (daily_df.mean() / daily_df.std())*np.sqrt(252)

        self.log_print(f'validation backtest sharpe: {sharpe}')
        #end backtest

        validation_return = np.concatenate(validation_return)
        validation_sharpe = (validation_return.mean() / validation_return.std() )* np.sqrt(252*24*(60/self.config['minute_predict']))
        self.log_print(f'validation sharpe: {validation_sharpe}')

        element_loss_df = pd.DataFrame(element_loss,columns=list(loss_dict.keys())).mean()
        self.log_print(f'validation element wise loss: {element_loss_df}')
        
        #return np.mean(loss_list)
        return -sharpe

    def trainning_epoch(self):
        epoch_loss = []
        element_loss = []

        self.model.train()
        if self.show:
            pbar = tqdm(self.trainning_dataloader, desc='Trainning',unit='batch')
        else:
            pbar = self.trainning_dataloader

        training_return = []

        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            label = batch['label']

            with torch.amp.autocast('cuda', dtype=self.mixed_percision_type):
                output_dict = self.model(batch)
                loss_dict = self.loss_fn(output_dict,label)
                loss = loss_dict['loss']

            self.scaler.scale(loss).backward()


            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad()

            trades = output_dict['trades']
            batch_return = (trades*label).mean(dim=1)

            if self.show:
                show_dict = {}

                for name,value in loss_dict.items():
                    show_dict[name] = value.item()

                pbar.set_postfix(show_dict)

            epoch_loss.append(loss.item())
            element_loss.append([i.item() for i in loss_dict.values()])


            training_return.append(batch_return.cpu().detach().numpy().reshape(-1))


        loss_valid = self.validation()

        training_return = np.concatenate(training_return)
        training_sharpe = (training_return.mean() / training_return.std() )* np.sqrt(252*24*(60/self.config['minute_predict']))
        self.log_print(f'training sharpe: {training_sharpe}')

        #get element wise loss
        element_loss_df = pd.DataFrame(element_loss,columns=list(loss_dict.keys())).mean()
        self.log_print(f'training element wise loss: {element_loss_df}')

        return np.mean(epoch_loss), loss_valid

    def train_main(self):
        best_loss = 1e7

        flat_count = 0
        for i in range(self.max_epcoh):

            train_loss, val_loss = self.trainning_epoch()
            self.trainning_loss_list.append(train_loss)


            self.log_print(f'epoch {i} loss {train_loss}; validation {i} loss {val_loss}')

            if self.scheduler_name is not None:
                self.scheduler.step()
                self.lr = self.scheduler.get_last_lr()
            self.log_print(f'current lr is :{self.lr}')


            torch.cuda.empty_cache()

            if val_loss < best_loss:
                torch.save(self.model.state_dict(), self.model_name)
                best_loss = val_loss
                flat_count = 0
            else:
                flat_count += 1

            if flat_count > self.exit_count:
                self.log_print('validation loss is not improving, stop')
                break

            gc.collect()
            self.validation_loss_list.append(val_loss)

        self.log_print('load model with best validation score')
        checkpoint = torch.load(self.model_name)
        self.model.load_state_dict(checkpoint)
        if not self.show:
            self.log_print(f'finish trainning, best valid score: {best_loss}')
        return best_loss
    

    def test(self):

        self.log_print('load model with best validation score for testing')
        checkpoint = torch.load(self.model_name)
        self.model.load_state_dict(checkpoint)

        final_df = []

        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(self.trading_dataloader, desc='Trading',unit='batch')

            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                timeint = batch['timeint']

                timeint_array = timeint[:,-1].cpu().numpy().reshape(-1)
                dateint = batch['dateint'][:,-1].cpu().numpy().reshape(-1)

                real_return = batch['real_return'].cpu().numpy().reshape(-1)

                output_dict = self.model(batch)
                trades = output_dict['trades'].cpu().numpy().reshape(-1)

                # print(trades)
                # print(timeint_array)
                # raise ValueError


                trading_df = pd.DataFrame({'trade':trades,
                                                    'real_return':real_return,
                                                    'dateint':dateint,
                                                    'trading_time':timeint_array,})
                # print(trading_df)
                # raise ValueError
                final_df.append(trading_df)

        final_df = pd.concat(final_df).reset_index(drop=True)

        final_df_copy = final_df.copy()

        final_df_copy = final_df_copy[(final_df_copy.trading_time < 170000)&(final_df_copy.trading_time > 90000)].reset_index(drop=True).copy()

        final_df_copy.loc[:,'vol_weight'] =  0.15 / (final_df_copy.real_return.rolling(self.config['std_lookback']).std() * np.sqrt(252*24*60))

        final_df_copy.loc[:,'raw_trade'] = final_df_copy.trade.rolling(self.config['minute_predict']).sum()

        final_df_copy.loc[:,'weight'] = final_df_copy.loc[:,'raw_trade'] * final_df_copy.loc[:,'vol_weight']

        final_df_copy.loc[:,'pnl'] = final_df_copy.loc[:,'weight']  * final_df_copy['real_return'].shift(-1)

        final_df_copy.loc[:,'pnl'].cumsum().plot()

        plt.savefig(self.backtest_name) 
        plt.close()
    
class Sharpe_LossV3(nn.Module):
    def __init__(self,
                 tau = 10,
                 balance_coeff = 0.1,
                 eps = 1e-11,
                ):
        '''
        this version predict a single step instead of multiple steps
        '''
        super().__init__()

        self.tau = tau
        self.balance_coeff = balance_coeff
        self.eps = eps

    def __call__(self, output_dict,label):
        trades = output_dict['trades']
        returns = trades * label
        mean_return = returns.mean()
        std = returns.std() + self.eps
        sharpe = (mean_return/std) * np.sqrt(252*8*3)
       
        
        balance = (self.balance_coeff*trades).sum().pow_(2)  + self.eps
        
        return {'loss':-self.tau*sharpe + balance, 
                'sp': -self.tau*sharpe,
                'balance':balance,
               }

def main(config_name, config, existing_model=None):
    try:
        print(config)
        database = Database(start_trainning_date=config['start_trainning_date'],
                 end_trainning_date=config['end_trainning_date'],
                 start_trading_date=config['start_trading_date'],
                 end_trading_date=config['end_trading_date'],
                 minute_predict=config['minute_predict'],
                 std_lookback=config['std_lookback'],
                 lookback=config['lookback_period'],
                 k=config['k'],
                 train_split=config['train_split'],
                 start_trading=133000,  # 
                 end_trading=213000,
                 contract_trading = 'es',
                 cv = config['cv'])


        training_dataloader,validation_dataloader,trading_dataloader = database.create_dataloader()
        if training_dataloader is None:
            print(f'No data found for {config["contract_trading"]}')
            return

        model = ITEncoder(d_model=config['d_model'],
                          nhead=config['nhead'],
                          layers=config['layers'], 
                          window_size=config['lookback_period'],
                          num_features=57,
                          ts_dim=config['ts_dim'],
                          num_filters=config['num_filters'])

        lr =config['lr']
        decay_rate = 0.00
        max_gradient_norm =config['max_gradient_norm']

        scaler = torch.cuda.amp.GradScaler(enabled= True)

        parameters = add_weight_decay(model,
                                    decay_rate,
                                    skip_list=["bias", "LayerNorm.bias"],  # , "LayerNorm.weight"],
                                        )

        optimizer = torch.optim.Adam(parameters, lr = lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_fn = Sharpe_LossV3(balance_coeff=config['balance_coeff'],
                                tau = config['tau'])

        directory_path = f'ML_results/{config["contract_trading"]}/{config_name}'
        os.makedirs(directory_path, exist_ok=True)

        if existing_model is not None:
            checkpoint = torch.load(f'{directory_path}/{existing_model}.pt')
            model.load_state_dict(checkpoint)

        model_name = f'{config["contract_trading"]}'

        trainner =  Trainner_V2(model = model,
                                optim = optimizer,
                                scaler = scaler,
                                device = device,
                                lr = lr,
                                loss_fn = loss_fn,
                                config = config,
                                trainning_dataloader = training_dataloader,
                                validation_dataloader = validation_dataloader,
                                trading_dataloader = trading_dataloader,
                                scheduler_name = config['scheduler_name'],
                                scheduler_max_step = 20,
                                scheduler_min_lr = 1e-9,
                                max_epoch = 100,
                                exit_count = 25,
                                model_name = f'{directory_path}/{model_name}_{config_name}.pt',
                                output_filaname = f'{directory_path}/{model_name}_{config_name}.txt',
                                backtest_name = f'{directory_path}/{model_name}_{config_name}.png',
                                show = True,
                                gradient_clip = config['gradient_clip'],
                                max_gradient_norm = max_gradient_norm,
                                gamma = 2,
                                poly_power = 2,
                                mixed_percision_type = torch.bfloat16
                                )

        try:
            # telegram_bot_sendtext(f'start: {folder_name};{config_file_name};{config_name}')
            # telegram_bot_sendtext('start: ')
            trainner.train_main()
            trainner.test()

        except Exception as e:
            traceback.print_exc()
            # telegram_bot_sendtext(e)
            trainner.test()
            return 0

        return 0
    except Exception as e:
        print(f'Error in {config["contract_trading"]}')
        print(traceback.format_exc())
        return

if __name__ == '__main__':
    products = ['es']
    current_dir = os.path.dirname(__file__)
    for product in products:
        # Load data from es_minutebar_data.csv using pandas
        config_path = os.path.join(current_dir, 'config', f'{product}.yaml')
        with open(config_path) as configFile:
            configs = yaml.load(configFile, Loader=yaml.FullLoader)
        for config_name, config in configs.items():
            #print(config)
            _ = main(config_name, config)

