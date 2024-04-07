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


def get_pos(factor, lookback_days, holding_days, threshold, coef_vol, volatility_adjust, exec_mode):
    # z-score
    factor_z_score = (factor - factor.rolling(lookback_days).mean()) / factor.rolling(lookback_days).std()
    # amount filter
    liquidity = get_liquidity("1440min")
    factor_z_score[~np.isnan(liquidity)] = np.nan
    weight_array = None
    if exec_mode == "longshort":
        # longshort, threshold used outside generate_longshort_weight
        factor_z_score[(factor_z_score < threshold) & (factor_z_score > -threshold)] = np.nan  # between -0.5, 0.5
        weight_array = generate_longshort_weight(factor_z_score.values, holding_days, GROUP_NUM)
    elif exec_mode == "timeseries":
        # timeseries, threshold used inside generate_timeseries_weight
        weight_array = generate_timeseries_weight(factor, coef_vol, holding_days, threshold, volatility_adjust)
    product_weight = pd.DataFrame(weight_array, index=factor.index, columns=factor.columns)
    product_weight = product_weight.fillna(0)  # in case of pos lost
    pos = product_weight.rolling(holding_days).sum()
    return pos


def generate_factor(historical_data):
    close = historical_data["close"]
    taker_volume = historical_data["taker_base_volume"]
    volume = historical_data["volume"]
    ret = np.log(close) - np.log(close.shift(1))
    factor = taker_volume / volume
    # factor = (taker_volume / volume) * np.sign(ret)
    return factor.shift(1)


if __name__ == '__main__':
    trade_uni = get_trade_uni()
    fea_lst = ["close", "close_BTC", "high", "low", "open", "quote_volume", "taker_base_volume", "taker_quote_volume",
               "trades", "volume"]
    freq = "1440min"
    historical_data = get_historical_data(freq, fea_lst)

    factor = generate_factor(historical_data)
    ret = np.log(historical_data["close"]) - np.log(historical_data["close"].shift(1))
    out = False
    if not out:
        ret = ret.loc[ret.index <= "2021-12-31"]
    factor = factor.reindex(index=ret.index)
    # parameters
    exec_mode = "longshort"
    lookback_days = 3
    holding_days = 1
    threshold = 0.5
    coef_vol = adj_vol_coef(ret, STD_WINDOW, TARGET_VOL)
    volatility_adjust = False
    out = False
    # single backtest   # core functions # longshort + timeseries
    pos = get_pos(factor, lookback_days, holding_days, threshold, coef_vol, volatility_adjust, exec_mode)
    print(f"{pos.loc[:, (pos != 0).any(axis=0)].shape[1]} products have positions")
    basecode_return = pos * ret
    bt_dict = backtest_stats(ret, pos, lookback_days, holding_days, threshold, exec_mode)
    pnl = bt_dict["pnl"]
    # top 10 pnl
    selected_products = basecode_return.cumsum().iloc[-1, :].sort_values(ascending=False).index[:10].tolist()
    plot_pnl_general(basecode_return, bt_dict["pos"], symbol="combo")
    plot_product_pnl_general(basecode_return[selected_products], 50)
    # plot_cumulative_pos_general(pos, "portfolio")
    print("debug point here")
    # above single version done
