import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from modules_importation import *
from modules_select_coin_backtest import *
import time


def single_param_backtest(ret, factor, lookback_days, holding_days, threshold, coef_vol, volatility_adjust, exec_mode):
    print(
        f"{log_info()} {exec_mode} [{factor_name}] lookback_days {lookback_days} holding_days "
        f"{holding_days} threshold {threshold} BEGINS")
    pos = get_pos(factor, lookback_days, holding_days, threshold, coef_vol, volatility_adjust, exec_mode)
    bt_dict = backtest_stats(ret, pos, lookback_days, holding_days, threshold, exec_mode)
    print(
        f"{log_info()} {exec_mode} [{factor_name}] lookback_days {lookback_days} holding_days "
        f"{holding_days} threshold {threshold} ENDS")
    return bt_dict


if __name__ == '__main__':
    trade_uni = get_trade_uni()
    fea_lst = ["close", "close_BTC", "high", "low", "open", "quote_volume", "taker_base_volume", "taker_quote_volume",
               "trades", "volume"]

    '''Parameters dict'''
    freq = "1440min"
    volatility_adjust = False
    exec_mode = "longshort"
    factor_name = "factor_er_ratio"
    lookback_days_range = np.arange(1, 30, 1)
    holding_days = 7  # timeseries adjust
    holding_days_range = np.arange(1, 30, 1)
    threshold = 0.5  # longshort adjust
    threshold_range = np.round(np.arange(0, 2, 0.1), 2)
    out = 0
    # data
    historical_data = get_historical_data(freq, fea_lst)
    # factor
    factor = run_factor_from_file(f"{factor_name}", "generate_factor", historical_data)
    ret = np.log(historical_data["close"]) - np.log(historical_data["close"].shift(1))
    if not out:
        ret = ret.loc[ret.index <= "2021-12-31"]
    factor = factor.reindex(index=ret.index)
    coef_vol = adj_vol_coef(ret, STD_WINDOW, TARGET_VOL)
    # mp backtest
    if exec_mode == "longshort":
        start = time.time()
        mp_res = Parallel(n_jobs=N_JOB, backend='loky')(
            delayed(single_param_backtest)(ret, factor, lookback_days, holding_days, threshold, coef_vol,
                                           volatility_adjust,
                                           exec_mode)
            for lookback_days in lookback_days_range
            for holding_days in holding_days_range)
        # mp_res = Parallel(n_jobs=N_JOB, backend='loky')(
        #     delayed(single_param_backtest)(ret, factor, lookback_days, holding_days, threshold, coef_vol,
        #                                    volatility_adjust,
        #                                    exec_mode)
        #     for lookback_days in lookback_days_range
        #     for holding_days in holding_days_range
        #     for threshold in threshold_range)
        end = time.time()
        print(f"============= Backtesting comsumes {round((end - start), 3)} s =============")
    else:
        start = time.time()
        mp_res = Parallel(n_jobs=N_JOB, backend='loky')(
            delayed(single_param_backtest)(ret, factor, lookback_days, holding_days, threshold, coef_vol,
                                           volatility_adjust,
                                           exec_mode)
            for lookback_days in lookback_days_range
            for threshold in threshold_range)
        end = time.time()
        print(f"============= Backtesting comsumes {round((end - start), 3)} s =============")

    # find best parameters
    stats_lst = []
    for i in range(len(mp_res)):
        stats_lst.append(mp_res[i]["stats"])
    all_stats = pd.concat(stats_lst, axis=1).T
    all_stats["lookback_days"] = pd.Series(all_stats.index).apply(lambda x: int(x.split("_")[0])).values
    all_stats["holding_days"] = pd.Series(all_stats.index).apply(lambda x: int(x.split("_")[1])).values
    all_stats["threshold"] = pd.Series(all_stats.index).apply(lambda x: float(x.split("_")[2])).values
    if exec_mode == "longshort":
        pivot_sr = pd.pivot_table(all_stats, values="sr", index="lookback_days", columns="holding_days")
    else:
        pivot_sr = pd.pivot_table(all_stats, values="sr", index="lookback_days", columns="threshold")

    best_param = all_stats['sr'].sort_values(ascending=False).index[0]
    best_lookback = int(best_param.split('_')[0])
    best_holding = int(best_param.split('_')[1])
    best_threshold = float(best_param.split('_')[2])

    # pnl plot
    print(f"-----> Geting the best triple param backtest result")
    bt_dict = single_param_backtest(ret, factor, best_lookback, best_holding, best_threshold, coef_vol,
                                    volatility_adjust, exec_mode)
    # check pos stats
    tmp_pos = bt_dict["pos"].fillna(0)
    print(f"-----> {tmp_pos.loc[:, (tmp_pos != 0).any(axis=0)].shape[1]} products covered positions")
    print(f"-----> cross-section max {np.max(tmp_pos.ne(0).sum(axis=1))} products")  # max
    print(f"-----> best param {best_param}")
    plot_heatmap_general(pivot_sr)
    # top 10 pnl
    top_num = 15
    selected_products = bt_dict["basecode_return"].cumsum().iloc[-1, :].sort_values(ascending=False).index[
                        :top_num].tolist()
    plot_pnl_general(bt_dict["basecode_return"], bt_dict["pos"], interval=50, best_param=best_param)
    plot_product_pnl_general(bt_dict["basecode_return"][selected_products])
    print("debug point here")
