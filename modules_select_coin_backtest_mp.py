import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from modules_generate_features import *
from modules_select_coin_backtest import *


def single_param_backtest(ret, factor, lookback_days, holding_days, threshold, coef_vol, volatility_adjust, exec_mode):
    print(f"{log_info()} lookback_days {lookback_days} holding_days {holding_days} threshold {threshold} BEGINS")
    pos = get_pos(factor, lookback_days, holding_days, threshold, coef_vol, volatility_adjust, exec_mode)
    bt_dict = backtest_stats(ret, pos, lookback_days, holding_days, threshold, exec_mode)
    print(f"{log_info()} lookback_days {lookback_days} holding_days {holding_days} threshold {threshold} ENDS")
    return bt_dict


if __name__ == '__main__':
    trade_uni = get_trade_uni()
    fea_lst = ["close", "close_BTC", "high", "low", "open", "quote_volume", "taker_base_volume", "taker_quote_volume",
               "trades", "volume"]

    '''Parameters dict'''
    freq = "1440min"
    volatility_adjust = False
    exec_mode = "timeseries"
    lookback_days_range = np.arange(1, 60, 1)
    holding_days = 1
    holding_days_range = np.arange(1, 60, 1)
    # threshold = 1.8
    threshold_range = np.round(np.arange(0, 3, 0.1), 2)
    out = 1
    '''Parameters dict'''

    historical_data = get_historical_data(freq, fea_lst)

    factor = generate_factor(historical_data)
    ret = np.log(historical_data["close"]) - np.log(historical_data["close"].shift(1))

    if not out:
        ret = ret.loc[ret.index <= "2021-12-31"]
    factor = factor.reindex(index=ret.index)
    coef_vol = adj_vol_coef(ret, STD_WINDOW, TARGET_VOL)

    mp_res = None
    if exec_mode == "longshort":
        mp_res = Parallel(n_jobs=N_JOB, backend='loky')(
            delayed(single_param_backtest)(ret, factor, lookback_days, holding_days, threshold, coef_vol,
                                           volatility_adjust,
                                           exec_mode)
            for lookback_days in lookback_days_range
            for holding_days in holding_days_range)
    elif exec_mode == "timeseries":
        mp_res = Parallel(n_jobs=N_JOB, backend='loky')(
            delayed(single_param_backtest)(ret, factor, lookback_days, holding_days, threshold, coef_vol,
                                           volatility_adjust,
                                           exec_mode)
            for lookback_days in lookback_days_range
            for threshold in threshold_range)
    # find best parameters
    stats_lst = []
    for i in range(len(mp_res)):
        stats_lst.append(mp_res[i]["stats"])
    all_stats = pd.concat(stats_lst, axis=1).T
    all_stats["lookback_days"] = pd.Series(all_stats.index).apply(
        lambda x: int(x.split("_")[0])).values
    if exec_mode == "longshort":
        all_stats["second"] = pd.Series(all_stats.index).apply(
            lambda x: int(x.split("_")[1])).values
    else:
        all_stats["second"] = pd.Series(all_stats.index).apply(
            lambda x: float(x.split("_")[1])).values
    pivot_sr = pd.pivot_table(all_stats, values="sr", index="lookback_days", columns="second")

    best_param = all_stats['sr'].sort_values(ascending=False).index[0]
    best_lookback = int(best_param.split('_')[0])
    if exec_mode == "longshort":
        best_second = int(best_param.split('_')[1])  # holding_days for ls
    else:
        best_second = float(best_param.split('_')[1])  # threshold for timeseries

    # pnl plot
    bt_dict = None
    if exec_mode == "longshort":
        bt_dict = single_param_backtest(ret, factor, best_lookback, best_second, threshold, coef_vol, volatility_adjust,
                                        exec_mode)
    elif exec_mode == "timeseries":
        bt_dict = single_param_backtest(ret, factor, best_lookback, holding_days, best_second, coef_vol,
                                        volatility_adjust,
                                        exec_mode)
    # check pos number
    tmp_pos = bt_dict["pos"].fillna(0)
    print(f"--- {tmp_pos.loc[:, (tmp_pos != 0).any(axis=0)].shape[1]} products covered positions")  # most for all time
    print(f"--- cross-section max {np.max(tmp_pos.ne(0).sum(axis=1))} products")  # max
    print(f"--- best param {best_param}")
    # top 10 pnl
    selected_products = bt_dict["basecode_return"].cumsum().iloc[-1, :].sort_values(ascending=False).index[:10].tolist()
    plot_pnl_general(bt_dict["basecode_return"], bt_dict["pos"], interval=50, best_param=best_param)
    plot_heatmap_general(pivot_sr)
    print("debug point here")
