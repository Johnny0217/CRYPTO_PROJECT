from datetime import datetime
import importlib
import pandas as pd
import numpy as np
from scipy.stats import norm

from numba import jit
import warnings
import os
import line_profiler

profile = line_profiler.LineProfiler()
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from numpy.lib import stride_tricks, pad
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')
# Global Variables
F_ZERO = config.getfloat('Backtest', 'F_ZERO')
FEE_RATE = config.getfloat('Backtest', 'FEE_RATE')
STD_WINDOW = config.getint('Backtest', 'STD_WINDOW')
TARGET_VOL = config.getfloat('Backtest', 'TARGET_VOL')
SIGNAL_LIMIT = config.getint('Backtest', 'SIGNAL_LIMIT')
N_JOB = config.getint('Backtest', 'N_JOB')
OUTLIER_STD = config.getint('Backtest', 'OUTLIER_STD')
GROUP_NUM = config.getint('Backtest', 'GROUP_NUM')
PROJECT_PATH = config['Backtest']['PROJECT_PATH']
TRADE_INTERVAL_IN_YEARS = config.getint('Backtest', 'TRADE_INTERVAL_IN_YEARS')


# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(1,1,1)
# interval = 50
# x = pos.index.astype(str)
# xticks = np.arange(0, len(x), interval)
# xticklabels = x[xticks]
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticklabels, rotation=90, fontsize=6)
# ax.set_title("")

def get_factor(factor_name, function_name, historical_data, params_dict):
    ''' This function is used to call factor.py file in another place
    :param factor_name:
    :param function_name:
    :param historical_data: used in generate factor
    :param params_dict: used in generate factor
    :return:
    '''
    args = (historical_data, params_dict)

    def load_factor(factor_name, function_name, *args, **kwargs):
        '''
        the structure of this function is stable as well as parameters
        :param args: used in generate factor
        '''
        module = importlib.import_module(f'factor.{factor_name}')
        func = getattr(module, function_name)
        return func(*args, **kwargs)

    factor = load_factor(factor_name, function_name, *args)
    return factor


def get_timestamp_interval_label(timestamps_series, freq):
    '''
    :param timestamps_series: datetime64[ns] "2023-12-31 18:59:00"
    :param freq: 1H 4H
    :return: interval label
    '''
    base_time = pd.to_datetime("00:00:00")
    time_diff = timestamps_series - base_time
    interval = freq
    labels = base_time + (time_diff // pd.Timedelta(interval)) * pd.Timedelta(interval)
    return labels


def get_1Dstrided_view(arr, window_size):
    return np.lib.stride_tricks.as_strided(arr, shape=(len(arr) - window_size + 1, window_size),
                                           strides=(arr.itemsize, arr.itemsize))


# not unique
def load_factor_py(factor_file, function, *args, **kwargs):
    '''
    :param args: used in find factor py file and run the factor function
    '''
    module = importlib.import_module(f'factor.{factor_file}')
    func = getattr(module, function)
    return func(*args, **kwargs)


# not unique
def get_factor_value(factor_file, function, params_dict, trade_uni, min_data_dict):
    args = (trade_uni, params_dict, min_data_dict)  # parameters used in that generate_factor function
    factor = load_factor_py(factor_file, function, *args)
    return factor


def backtest_stats(return_data, basecode_pos, lookback_days, holding_days, threshold, exec_mode, is_long_only=False):
    if is_long_only:
        basecode_pos[basecode_pos < 0] = 0
    basecode_return = (basecode_pos * return_data).fillna(.0)
    portfolio_pnl = basecode_return.sum(axis=1).cumsum()
    stats = get_stats(basecode_return.sum(axis=1), basecode_return.sum(axis=1), basecode_pos)
    res_dict = {}
    res_dict['basecode_return'] = basecode_return
    res_dict['stats'] = pd.Series(stats)
    res_dict['pnl'] = portfolio_pnl
    res_dict['pos'] = basecode_pos
    label = None
    if exec_mode == "longshort":
        label = str(lookback_days) + '_' + str(holding_days)
    elif exec_mode == "timeseries":
        label = str(lookback_days) + '_' + str(threshold)
    for key in res_dict.keys():
        res_dict[key].name = label
    return res_dict


def adj_vol_coef(return_data, STD_WINDOW, TARGET_VOL):
    '''
    used twice
    '''
    trade_intervals_in_year = 360
    ret_std = return_data.rolling(STD_WINDOW).std().shift() * np.sqrt(trade_intervals_in_year)
    coef = TARGET_VOL / ret_std
    return coef


@jit
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


# @jit
# def generate_longshort_weight(factor_value_array, holding_days: int, group_num: int):
#     '''
#     version2 long-short 22 33 55 55 ... fixed number of codes
#     '''
#     weight_array = np.zeros(factor_value_array.shape)
#     for j in range(factor_value_array.shape[0]):
#         pre_value = factor_value_array[j, :]  # give pos line by line
#         product_num = len(pre_value[~np.isnan(pre_value)])  # num of product has factors
#         if product_num > 0:  # in cross-section, it has factors, need to allocate pos
#             num_long = num_short = round(product_num / 2)  # 1 / 20
#             if product_num > 10:
#                 num_long = num_short = 5
#             product_long_index = np.where(pre_value >= -np.sort(-pre_value)[num_long - 1])
#             product_short_index = np.where(pre_value <= np.sort(pre_value)[num_short - 1])
#             nan_in_long_factor_value = np.isnan(pre_value[product_long_index]).sum()  # top long value
#             nan_in_short_factor_value = np.isnan(pre_value[product_short_index]).sum()  # top short value
#             if nan_in_long_factor_value < num_long and nan_in_short_factor_value < num_short:
#                 weight_array[j, product_long_index] = (1 / len(product_long_index[0])) / holding_days
#                 weight_array[j, product_short_index] = -(1 / len(product_short_index[0])) / holding_days
#     return weight_array


# @jit
# def generate_longshort_weight(factor_value_array, holding_days: int, group_num: int):
#     '''
#     Version3 lo max products in a cross-section: 5 products lo only
#     '''
#     weight_array = np.zeros(factor_value_array.shape)
#     for j in range(factor_value_array.shape[0]):
#         a = j
#         pre_value = factor_value_array[j, :]  # factor value in factor_value_array line x
#         product_num = len(pre_value[~np.isnan(pre_value)])  # num of product has factors
#         if product_num > 0:  # has factors in cross-section
#             num_long = product_num
#             if product_num > 5:
#                 num_long = 5  # number of long positions
#             product_long_index = np.where(pre_value >= -np.sort(-pre_value)[num_long - 1])  # long pos location
#             nan_in_long_factor_value = np.isnan(pre_value[product_long_index]).sum()  # maybe do not need
#             if nan_in_long_factor_value < num_long:  # always True
#                 weight_array[j, product_long_index] = (1 / len(product_long_index[0])) / holding_days
#     return weight_array


# def get_longshort_pos(factor, lookback_days, holding_days, pos_demean=0):
#     # CTA version
#     factor_value = factor.rolling(lookback_days).mean() / factor.rolling(lookback_days).std()
#     # factor_value_winsorize = winsorize(factor_value)
#     factor_value_winsorize = factor_value.copy()
#     weight_array = generate_longshort_weight(factor_value_winsorize.values, holding_days, GROUP_NUM)
#     product_weight = pd.DataFrame(weight_array, index=factor.index, columns=factor.columns)
#     product_weight = product_weight.fillna(0)  # in case of pos lost
#     pos = product_weight.rolling(holding_days).sum()
#     if pos_demean > 0:
#         pos = pos - pos.rolling(pos_demean).mean()
#     pos = pos.fillna(.0)
#     return pos

def get_liquidity(freq):
    amount_path = os.path.join(PROJECT_PATH, "binance-feature", freq, "quote_volume.csv")
    amount = pd.read_csv(amount_path, index_col=0)
    close_path = os.path.join(PROJECT_PATH, "binance-feature", freq, "close.csv")
    close = pd.read_csv(close_path, index_col=0)
    volume_path = os.path.join(PROJECT_PATH, "binance-feature", freq, "volume.csv")
    volume = pd.read_csv(volume_path, index_col=0)
    amount_substitute = close * volume
    combined_amount = amount.combine_first(amount_substitute)
    liquidity = combined_amount.rolling(7).mean()
    liquidity[liquidity < 2000 * 10000] = np.nan
    return liquidity.shift(1)


def get_longshort_pos(factor, lookback_days, holding_days, threshold, pos_demean=0):
    # z-score
    factor_z_score = (factor - factor.rolling(lookback_days).mean()) / factor.rolling(lookback_days).std()
    # amount filter
    liquidity = get_liquidity("1440min")
    factor_z_score[~np.isnan(liquidity)] = np.nan
    # threshold
    factor_z_score[(factor_z_score < threshold) & (factor_z_score > -threshold)] = np.nan  # between -1, 1
    # 3 groups, long-short,
    weight_array = generate_longshort_weight(factor_z_score.values, holding_days, GROUP_NUM)
    product_weight = pd.DataFrame(weight_array, index=factor.index, columns=factor.columns)
    product_weight = product_weight.fillna(0)  # in case of pos lost
    pos = product_weight.rolling(holding_days).sum()
    if pos_demean > 0:
        pos = pos - pos.rolling(pos_demean).mean()
    pos = pos.fillna(.0)
    return pos


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


def get_timeseries_pos(factor, return_data, lookback_days, holding_days, threshold, volatility_adjust, pos_demean=0):
    '''
    used twice
    '''
    # z-score
    factor_value = (factor - factor.rolling(lookback_days).mean()) / factor.rolling(
        lookback_days).std()
    # factor_value[abs(factor_value) > OUTLIER_STD] = np.nan      # check Dr.J
    # factor_value[factor_value < threshold] = np.nan
    coef_vol = adj_vol_coef(return_data, STD_WINDOW, TARGET_VOL)
    weight_array = generate_timeseries_weight(factor_value, coef_vol, holding_days,
                                              threshold, volatility_adjust)
    product_weight = pd.DataFrame(weight_array, index=factor.index, columns=factor.columns)
    product_weight = product_weight.fillna(0)
    pos = product_weight.rolling(holding_days).sum()
    if pos_demean > 0:
        pos = pos - pos.rolling(pos_demean).mean()
    pos = pos.fillna(.0)
    return pos


def log_info():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_annual_return(return_array):
    trade_intervals_in_year = 360
    return np.nanmean(return_array) * trade_intervals_in_year


def get_annual_volatility(return_array):
    # trade_intervals_in_year = int(1440 / freq_int) * 360
    trade_intervals_in_year = 360
    return np.nanstd(return_array) * np.sqrt(trade_intervals_in_year)


def get_maxdrawdown(return_array):
    # actual mdd
    return_array = return_array.dropna()
    max_dd = (return_array.cumsum().cummax() - return_array.cumsum()).max()
    # relative mdd
    # pnl = return_array.cumsum() + 1
    # cum_max = pnl.cummax()
    # dd = (pnl - cum_max) / cum_max
    # max_dd = abs(dd).max()
    return max_dd


def get_sharpe_ratio(return_array):
    annual_return = get_annual_return(return_array)
    annual_volatility = get_annual_volatility(return_array)
    return annual_return / annual_volatility


def get_calmar_ratio(return_array):
    annual_return = get_annual_return(return_array)
    max_drawdown = get_maxdrawdown(return_array)
    return annual_return / max_drawdown


def get_daily_turnover(product_weight_df):
    average_interval_tvr = np.nanmean(abs(product_weight_df - product_weight_df.shift()).sum(axis=1))
    trade_intervals_in_day = 1
    return average_interval_tvr * trade_intervals_in_day


def get_return_in_bps(return_array_no_cost, product_weight_df):
    annual_return = get_annual_return(return_array_no_cost)
    daily_turnover = get_daily_turnover(product_weight_df)
    daily_return_in_bps = annual_return / (360 * daily_turnover)
    return daily_return_in_bps


def get_stats(return_series_cost, return_series_no_cost, pos_series):
    '''
    :param return_series_cost: one column, portfolio [array]
    :param return_series_no_cost: one column, portfolio [array]
    :param pos_series: all symbol pos [matrix]
    :return:
    '''
    stats = {}
    stats['annVol'] = get_annual_volatility(return_series_cost)
    stats['annRet'] = get_annual_return(return_series_cost)
    stats['calmar'] = get_calmar_ratio(return_series_cost)
    stats['sr'] = get_sharpe_ratio(return_series_cost)
    stats['turn'] = get_daily_turnover(pd.DataFrame(pos_series))
    stats['retinbps'] = get_return_in_bps(return_series_no_cost, pd.DataFrame(pos_series))
    stats['mdd'] = get_maxdrawdown(return_series_cost)
    return stats


def check_path(path):
    '''
    :param path: need to generate in project
    :return: generated path
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        return


def rolling_spearman(seq_a, seq_b, window):
    stride_a = seq_a.strides[0]
    set_array_a = stride_tricks.as_strided(seq_a,
                                           shape=[len(seq_a) - window + 1, window],
                                           strides=[stride_a, stride_a])
    stride_b = seq_b.strides[0]
    set_array_b = stride_tricks.as_strided(seq_b,
                                           shape=[len(seq_b) - window + 1, window],
                                           strides=[stride_b, stride_b])
    df_a = pd.DataFrame(set_array_a)
    df_b = pd.DataFrame(set_array_b)
    # more complex, but faster
    df_rank_a = df_a.rank(axis=1, pct=True)
    df_rank_b = df_b.rank(axis=1, pct=True)
    corr_df = df_rank_a.corrwith(df_rank_b, 1)
    # more simple, but slower
    # corr_df = df_a.corrwith(df_b. axis=1, method="spearman)
    return pad(corr_df, (window - 1, 0), 'constant', constant_values=np.nan)


def winsorize(factor_data, abnormal_ratio=0.625):
    '''
    :param factor_data: need to modify to pure numpy array
    :param abnormal_ratio:
    :return:
    '''
    k_mad = 1 / norm(0, 1).ppf(abnormal_ratio)

    # factor_data_median = pd.DataFrame(data=np.nan, index=factor_data.index, columns=factor_data.columns)
    # for col in factor_data.columns:
    #     factor_data_median[col] = factor_data.median(axis=1)

    factor_data_median = pd.DataFrame(np.tile(factor_data.median(axis=1), (len(factor_data.columns), 1))).T
    factor_data_median.index = factor_data.index
    factor_data_median.columns = factor_data.columns

    factor_data_mad1 = abs(factor_data - factor_data_median)
    # factor_data_mad2 = pd.DataFrame(data=np.nan, index=factor_data.index, columns=factor_data.columns)
    # for col in factor_data.columns:
    #     factor_data_mad2[col] = factor_data_mad1.median(axis=1)

    factor_data_mad2 = pd.DataFrame(np.tile(factor_data_mad1.median(axis=1), (len(factor_data.columns), 1))).T
    factor_data_mad2.index = factor_data.index
    factor_data_mad2.columns = factor_data.columns

    factor_inf = factor_data_median - 3 * k_mad * factor_data_mad2
    factor_sup = factor_data_median + 3 * k_mad * factor_data_mad2
    factor_data[factor_data < factor_inf] = factor_inf
    factor_data[factor_data > factor_sup] = factor_sup

    res = pd.DataFrame(data=np.nan, index=factor_data.index, columns=factor_data.columns)
    for col in res.columns:
        res[col] = (factor_data[col] - factor_data.mean(axis=1)) / factor_data.std(axis=1)
    return res


def plot_pos_general(pos, symbol=None, interval=50):
    '''
    :param pos: fillna(0) series
    :param interval: 50 int
    :param symbol: BTCUSDT str
    '''
    pos = pd.DataFrame(pos)
    pos = pos.sum(axis=1).fillna(0)
    x = pos.index.astype(str)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    xtick = np.arange(0, len(x), interval)
    xticklabel = pd.Series(x[xtick])
    ax.set_xticks(xtick)
    ax.set_xticklabels(xticklabel, rotation=90, fontsize=6)
    ax.set_title(f"Symbol {symbol} Position Distribution")
    ax.plot(x, pos.values)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.show()


def plot_cumulative_pos_general(pos, symbol=None, interval=50):
    pos = pd.DataFrame(pos)
    port_pos = pos.sum(axis=1).fillna(0)
    cum_port_pos = port_pos.cumsum()
    x = pos.index.astype(str)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    xtick = np.arange(0, len(x), interval)
    xticklabel = pd.Series(x[xtick])
    ax.set_xticks(xtick)
    ax.set_xticklabels(xticklabel, rotation=90, fontsize=6)
    ax.set_title(f"Symbol {symbol} Cumulative Position")
    ax.plot(x, cum_port_pos.values, label="portfolio")
    for key in pos.columns:
        ax.plot(x, pos[key].cumsum().values, label=f"{key}")
    ax.legend(loc="best")
    plt.axhline(y=0, color='black', linestyle='-')
    plt.show()


def plot_ls_pnl_general(ret, pos, symbol=None, interval=50):
    long_pos = pos[pos > 0].fillna(0)
    short_pos = pos[pos < 0].fillna(0)
    long_pnl = (ret * long_pos).sum(axis=1).cumsum()
    short_pnl = (ret * short_pos).sum(axis=1).cumsum()
    x = pos.index.astype(str)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    xtick = np.arange(0, len(x), interval)
    xticklabel = pd.Series(x[xtick])
    ax.set_xticks(xtick)
    ax.set_xticklabels(xticklabel, rotation=90, fontsize=6)
    ax.set_title(f"Symbol {symbol} long short pnl")
    ax.plot(x, long_pnl, label="long", color="red")
    ax.plot(x, short_pnl, label="short", color="green")
    ax.legend(loc="best")
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axvline(x='2021-12-31', color='r', linestyle='--')
    plt.show()


def plot_heatmap_general(pivot_table):
    sns.heatmap(data=pivot_table, square=True, cmap="RdBu_r")
    plt.show()


def plot_product_pnl_general(basecode_return, interval=50):
    basecode_return = pd.DataFrame(basecode_return)
    x = basecode_return.index.astype(str)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    xtick = np.arange(0, len(x), interval)
    xticklabel = pd.Series(x[xtick])
    ax.set_xticks(xtick)
    ax.set_xticklabels(xticklabel, rotation=90, fontsize=6)
    ax.set_title("Products pnl")
    for symbol in basecode_return.columns:
        pnl = (basecode_return[symbol]).cumsum()
        ax.plot(x, pnl.values, label=f"{symbol} pnl")
    ax.legend(loc=0)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axvline(x='2021-12-31', color='r', linestyle='--')
    plt.show()


def plot_pnl_general(basecode_return, pos, symbol=None, description=None, interval=50, best_param=None):
    basecode_return = pd.DataFrame(basecode_return)
    pnl = basecode_return.sum(axis=1).cumsum().fillna(0)
    pos = pos.fillna(0)
    x = pos.index.astype(str)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    xtick = np.arange(0, len(x), interval)
    xticklabel = pd.Series(x[xtick])
    ax.set_xticks(xtick)
    ax.set_xticklabels(xticklabel, rotation=90, fontsize=6)
    stats = get_stats(pnl.diff(), pnl.diff(), pos)
    stats = pd.Series(stats)
    ax.set_title(
        f"{symbol}|{best_param}|SR={round(stats['sr'], 2)}|MDD={round(stats['mdd'], 2)}|"
        f"Calmar={round(stats['calmar'], 2)}|Ret={round(stats['annRet'], 2)}|Vol={round(stats['annVol'], 2)}"
        f"|Turn={round(stats['turn'], 4)}|Bps={round(stats['retinbps'], 4)}")
    ax.plot(x, pnl.values, label="pnl", color="orange", linewidth=2)
    ax.legend(loc=0)
    plt.axhline(y=0, color='black', linestyle='-')
    mdd_series = pd.Series(pnl.values) - pd.Series(pnl.values).cummax()
    y2 = mdd_series
    ax1 = ax.twinx()
    ax1.set_xticks(xtick)
    ax1.set_xticklabels(xticklabel, rotation=90, fontsize=6)
    ax1.plot(x, y2, label="MDD", alpha=0.3, color='g')
    ax1.fill_between(x, y2, alpha=0.3, color="g")
    plt.axvline(x='2021-12-31', color='r', linestyle='--')
    plt.show()
    # plt.savefig(f"{PROJECT_PATH}/{description}.jpeg", bbox_inches="tight")

def plot_longshort_product_bar_general(pos):
    interval = 50
    long_pos = np.sign(pos[pos > 0]).sum(axis=1)
    short_pos = np.sign(pos[pos < 0]).sum(axis=1) * -1
    x = pos.index.astype(str)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    xtick = np.arange(0, len(x), interval)
    xticklabel = pd.Series(x[xtick])
    ax.set_xticks(xtick)
    ax.set_xticklabels(xticklabel, rotation=90, fontsize=6)
    ax.set_title(f"long-short product number distribution")
    bar_width = 1
    ax.bar(x, long_pos.values, bar_width, label="long")
    ax.bar(x, short_pos.values, bar_width, label="short", bottom=long_pos.values)
    plt.legend(loc="best")
    plt.axhline(y=2, color="red", linestyle="--")
    plt.axhline(y=4, color="red", linestyle="--")
    plt.show()

def find_local_minima(series):
    local_minima = pd.Series(index=series.index)
    for i in range(1, len(series) - 1):
        if series[i] < series[i - 1] and series[i] < series[i + 1]:
            local_minima[series.index[i]] = series[i]
        else:
            local_minima[series.index[i]] = np.nan
    return local_minima
