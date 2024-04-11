import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def get_historical_data(freq, feature_lst):
    historical_data = {}
    for feature in feature_lst:
        print(f"{log_info()} {freq} {feature} data loaded")
        read_path = os.path.join(PROJECT_PATH, "data", "binance-feature", freq, f"{feature}.csv")
        historical_data[feature] = pd.read_csv(read_path, index_col=0)
    return historical_data


def get_liquidity(freq):
    amount_path = os.path.join(PROJECT_PATH, "data", "binance-feature", freq, "quote_volume.csv")
    amount = pd.read_csv(amount_path, index_col=0)
    close_path = os.path.join(PROJECT_PATH, "data", "binance-feature", freq, "close.csv")
    close = pd.read_csv(close_path, index_col=0)
    volume_path = os.path.join(PROJECT_PATH, "data", "binance-feature", freq, "volume.csv")
    volume = pd.read_csv(volume_path, index_col=0)
    amount_substitute = close * volume
    combined_amount = amount.combine_first(amount_substitute)
    liquidity = combined_amount.rolling(7).mean()
    liquidity[liquidity < 2000 * 10000] = np.nan
    return liquidity.shift(1)


def get_trade_uni():
    path = os.path.join(PROJECT_PATH, "data", "coin_binance_usdt.csv")
    df = pd.read_csv(path, index_col=0)
    trade_uni = df.iloc[:, 0].values.tolist()
    return trade_uni


if __name__ == '__main__':
    trade_uni = get_trade_uni()
    fea_lst = ["close", "close_BTC", "high", "low", "open", "quote_volume", "taker_base_volume", "taker_quote_volume",
               "trades", "volume"]
    freq = "1440min"
    historical_data = get_historical_data(freq, fea_lst)
