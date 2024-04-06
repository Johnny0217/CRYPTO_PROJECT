import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from modules_generate_features import *


def get_historical_data(freq, feature_lst):
    historical_data = {}
    for feature in feature_lst:
        print(f"{log_info()} {freq} {feature} data loaded")
        historical_data[feature] = pd.read_csv(f"{PROJECT_PATH}/data/binance-feature/{freq}/{feature}.csv",
                                               index_col=0)
    return historical_data


def generate_longshort_weight(factor_value_array, holding_days: int, group_num: int):
    # original version group_num -> GROUP_NUM
    weight_array = np.zeros(factor_value_array.shape)
    for j in range(factor_value_array.shape[0]):
        pre_value = factor_value_array[j, :]  # give pos line by line
        product_num = len(pre_value[~np.isnan(pre_value)])  # num of symbols -> factors
        if product_num > 0:  # in cross-section, it has factors, need to allocate pos
            num_long = num_short = round(product_num / group_num)  # 1 / 3
            product_long_index = np.where(pre_value >= -np.sort(-pre_value)[num_long - 1])
            product_short_index = np.where(pre_value <= np.sort(pre_value)[num_short - 1])
            nan_in_long_factor_value = np.isnan(pre_value[product_long_index]).sum()  # top long value
            nan_in_short_factor_value = np.isnan(pre_value[product_short_index]).sum()  # top short value
            if nan_in_long_factor_value < num_long and nan_in_short_factor_value < num_short:
                weight_array[j, product_long_index] = (1 / len(product_long_index[0])) / holding_days
                weight_array[j, product_short_index] = -(1 / len(product_short_index[0])) / holding_days
    return weight_array


def generate_timeseries_weight(factor, coef_vol, holding_days, threshold, volatility_adjust):
    '''
    used twice
    '''
    signal_df = pd.DataFrame(np.zeros(factor.shape), index=factor.index, columns=factor.columns)
    signal_df[factor > threshold] = 1
    signal_df[factor < -threshold] = -1
    if volatility_adjust:
        signal_df = signal_df * coef_vol
    signal_df[signal_df > SIGNAL_LIMIT] = SIGNAL_LIMIT
    signal_df[signal_df < -SIGNAL_LIMIT] = -SIGNAL_LIMIT
    signal_array = np.asarray(signal_df)
    weight_array = np.zeros(signal_array.shape)
    for i in range(signal_array.shape[0]):
        pre_value = signal_array[i, :]
        notnan_num = len(pre_value[~np.isnan(pre_value)])  # num of factor value
        if notnan_num >= 1:  # at least one factor value occured in cross_section
            divide_coef = notnan_num * holding_days
            notnan_loction = np.where(~np.isnan(pre_value))
            weight_array[i, notnan_loction] = signal_array[i, notnan_loction] / divide_coef
    return weight_array


if __name__ == '__main__':
    trade_uni = get_trade_uni()
    fea_lst = ["close", "close_BTC", "high", "low", "open", "quote_volume", "taker_base_volume", "taker_quote_volume",
               "trades", "volume"]
    freq = "1440min"
    historical_data = get_historical_data(freq, fea_lst)


    def generate_factor(historical_data, freq):
        close = historical_data["close"]
        ret = np.log(close) - np.log(close.shift(1))
        factor = ret.copy()
        return factor.shift(1)


    factor = generate_factor(historical_data, freq)
    ret = np.log(historical_data["close"]) - np.log(historical_data["close"].shift(1))
    lookback = 3
    holding = 1
    threshold = 0


    def get_pos(factor, lookback, holding, threshold):
        pass


    pos = get_pos(factor, lookback, holding, threshold)
    # after z-score, filtering liquidity, filtering happened before pos
    print("debug point here")
