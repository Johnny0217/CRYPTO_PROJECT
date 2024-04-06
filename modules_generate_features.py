import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def get_trade_uni():
    df = pd.read_csv(f"{PROJECT_PATH}/data/coin_binance_usdt.csv", index_col=0)
    trade_uni = df.iloc[:, 0].values.tolist()
    return trade_uni


def get_liquidity(freq):
    amount = pd.read_csv(f"{PROJECT_PATH}/data/binance-feature/{freq}/quote_volume.csv", index_col=0)
    close = pd.read_csv(f"{PROJECT_PATH}/data/binance-feature/{freq}/close.csv", index_col=0)
    volume = pd.read_csv(f"{PROJECT_PATH}/data/binance-feature/{freq}/volume.csv", index_col=0)
    amount_substitute = close * volume
    combined_amount = amount.combine_first(amount_substitute)
    liquidity = combined_amount.rolling(7).mean()
    liquidity[liquidity < 2000 * 10000] = np.nan
    return liquidity


if __name__ == '__main__':
    trade_uni = get_trade_uni()
    get_liquidity("1440min")

    print("debug point here")
