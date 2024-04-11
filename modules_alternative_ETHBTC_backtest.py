import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

'''
Archive py script, used for ETH BTC only
read only
'''

def load_product_feature_matrix(symbol, basic_features, alternative_features, out):
    data = pd.read_parquet(
        (os.path.join(PROJECT_PATH, "data", "binance-public-data", "1440min", f"{symbol}_1440m.parq")))
    data = data[basic_features]
    data.index = data.index.astype(str)
    data["return"] = np.log(data["close"]) - np.log(data["close"].shift(1))
    for feature in alternative_features:
        alter_df = pd.read_csv(os.path.join(PROJECT_PATH, "data", "bitinfocharts", f"{feature}.csv"), index_col=0)
        alter_df = alter_df.reindex(index=data.index)
        data[f"{feature}"] = alter_df[f"{symbol.replace('USDT', '').lower()}"]
    if not out:
        data = data.loc[data.index <= "2021-12-31"]
    return data


# def generate_factor(historical_data):
#     address = historical_data["activeaddresses"]
#     ret = historical_data["return"]
#     tweets = historical_data["tweets"]
#     num_tran_in_blockchain_per_day = historical_data["transactions"]
#     factor = num_tran_in_blockchain_per_day.rolling(28).skew() * np.sign(ret)
#     return factor.shift(1)

def generate_factor(historical_data):
    address = historical_data["activeaddresses"]
    ret = historical_data["return"]
    tweets = historical_data["tweets"]
    difficulty = historical_data["difficulty"]
    num_tran_in_blockchain_per_day = historical_data["transactions"]

    factor = difficulty.rolling(14).kurt()
    # factor = num_tran_in_blockchain_per_day.rolling(35).corr(ret, method="spearman")
    return factor.shift(1)


if __name__ == '__main__':
    trade_uni = ["ETHUSDT", "BTCUSDT"]
    # trade_uni = ["ETHUSDT"]
    out = False
    alternative_features = ["activeaddresses", "transactions", "difficulty", "fee_to_reward", "tweets", "hashrate",
                            "transactions"]
    basic_features = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_base_volume",
                      "taker_quote_volume"]

    symbol_data = {}
    for symbol in trade_uni:
        symbol_data[symbol] = load_product_feature_matrix(symbol, basic_features, alternative_features, out)
    selected_features = ["activeaddresses", "transactions", "difficulty", "fee_to_reward", "tweets", "open", "high",
                         "low", "close", "volume", "quote_volume", "transactions",
                         "trades", "taker_base_volume", "taker_quote_volume", "return", "hashrate"]
    historical_data = {}
    for feature in selected_features:
        data_lst = []
        for key in symbol_data.keys():
            data = symbol_data[key][feature]
            data.name = key
            data_lst.append(data)
        historical_data[feature] = pd.concat(data_lst, axis=1)

    # factor
    factor = generate_factor(historical_data)


    # single
    # pos = get_timeseries_pos(factor, historical_data["return"], lookback_days=53, holding_days=1, threshold=1,
    # volatility_adjust=False)
    # bt_dict = backtest_stats(historical_data["return"], pos, lookback_days=53, holding_days=1, is_long_only=False)
    # plot
    # plot_ls_pnl_general(historical_data["return"], pos, symbol=f'{trade_uni}')
    # plot_pos_general(pos, symbol=f'{trade_uni}')
    # plot_cumulative_pos_general(pos, symbol=f'{trade_uni}')

    # mp
    def single_param_bt(factor, return_data, lookback_days, holding_days, threshold, volatility_adjust, is_long_only):
        print(f"{log_info()} lookback {lookback_days} threshold {threshold} BEGINS!")
        pos = get_timeseries_pos(factor, return_data, lookback_days, holding_days, threshold, volatility_adjust)
        bt_dict = backtest_stats(return_data, pos, lookback_days, holding_days, threshold, is_long_only)
        print(f"{log_info()} lookback {lookback_days} threshold {threshold} ENDS!")
        return bt_dict


    lookback_arange = np.arange(1, 120, 2)
    threshold_arange = np.round(np.arange(0, 3, 0.1), 2)
    print(f"number of cores {os.cpu_count()}")
    mp_res = Parallel(n_jobs=10, backend='loky')(
        delayed(single_param_bt)(factor, historical_data["return"], lookback_days, 1, threshold,
                                 volatility_adjust=False,
                                 is_long_only=False) for lookback_days in lookback_arange for threshold in
        threshold_arange)
    stats_lst = []
    for i in range(len(mp_res)):
        stats_lst.append(mp_res[i]["stats"])
    res_bt = pd.concat(stats_lst, axis=1).T
    res_bt = res_bt.sort_values(by="sr", ascending=False)
    res_bt["lookback_index"] = pd.Series(res_bt.index).apply(lambda x: int(x.split("_")[0])).values
    res_bt["threshold_index"] = pd.Series(res_bt.index).apply(lambda x: x.split("_")[1]).values
    # heat amp
    heat_df = pd.pivot_table(res_bt, values=["sr"], index="lookback_index", columns=["threshold_index"])
    best_lookback = res_bt.index[0].split('_')[0]
    best_threshold = res_bt.index[0].split('_')[1]
    print(
        f"====== [best_param] lookback {res_bt.index[0].split('_')[0]} threshold {res_bt.index[0].split('_')[1]} ======")

    res = single_param_bt(factor, historical_data["return"], lookback_days=int(res_bt.index[0].split('_')[0]),
                          holding_days=1, threshold=float(res_bt.index[0].split('_')[1]), volatility_adjust=False,
                          is_long_only=False)

    plot_pnl_general(res["basecode_return"], res["pos"], symbol="combo", best_param=f'{best_lookback}_{best_threshold}')
    plot_heatmap_general(heat_df)
    # plot_product_pnl_general(res["basecode_return"], interval=50)
    # plot_cumulative_pos_general(res["pos"], symbol)
    print("debug point here")
