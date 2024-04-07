import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def get_trade_uni():
    path = os.path.join(PROJECT_PATH, "coin_binance_usdt.csv")
    df = pd.read_csv(path, index_col=0)
    trade_uni = df.iloc[:, 0].values.tolist()
    return trade_uni


if __name__ == '__main__':
    trade_uni = get_trade_uni()
    get_liquidity("1440min")

    print("debug point here")
