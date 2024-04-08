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
def run_factor_from_file(file_name, function_name, *args, **kwargs):
    module = importlib.import_module(f'factors.{file_name}')
    func = getattr(module, function_name)
    return func(*args, **kwargs)


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
    if threshold is not None:
        label = str(lookback_days) + '_' + str(holding_days) + '_' + str(threshold)
    for key in res_dict.keys():
        res_dict[key].name = label
    return res_dict


def adj_vol_coef(return_data, STD_WINDOW, TARGET_VOL):
    trade_intervals_in_year = 360
    ret_std = return_data.rolling(STD_WINDOW).std().shift() * np.sqrt(trade_intervals_in_year)
    coef = TARGET_VOL / ret_std
    return coef


def log_info():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_annual_return(return_array):
    trade_intervals_in_year = 360
    return np.nanmean(return_array) * trade_intervals_in_year


def get_annual_volatility(return_array):
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


def find_local_minima(series):
    local_minima = pd.Series(index=series.index)
    for i in range(1, len(series) - 1):
        if series[i] < series[i - 1] and series[i] < series[i + 1]:
            local_minima[series.index[i]] = series[i]
        else:
            local_minima[series.index[i]] = np.nan
    return local_minima


# ==================================================================================================================== #

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


def plot_cumulative_pos_general(pos, symbol=None, description=None, interval=50):
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
    save_path = os.path.join(PROJECT_PATH, "factors", f"{description}_cumpos.jpeg")
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()


def plot_ls_pnl_general(ret, pos, symbol=None, description=None, interval=50):
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
    save_path = os.path.join(PROJECT_PATH, "factors", f"{description}_ls_pnl.jpeg")
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()


def plot_heatmap_general(pivot_table, description=None):
    sns.heatmap(data=pivot_table, square=True, cmap="RdBu_r")
    save_path = os.path.join(PROJECT_PATH, "factors", f"{description}_heatmap.jpeg")
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()


def plot_product_pnl_general(basecode_return, description=None, interval=50):
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
    save_path = os.path.join(PROJECT_PATH, "factors", f"{description}_product_pnl.jpeg")
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()


def plot_pnl_general(basecode_return, pos, symbol=None, description=None, best_param=None, interval=50):
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
    save_path = os.path.join(PROJECT_PATH, "factors", f"{description}_pnl.jpeg")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


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
