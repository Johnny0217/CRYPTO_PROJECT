import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from modules_importation import *
from modules_op import *


def generate_factor(historical_data):
    close = historical_data["close"]
    high = historical_data["high"]
    low = historical_data["low"]
    open = historical_data["open"]
    amount = historical_data["quote_volume"]
    taker_volume = historical_data["taker_base_volume"]
    volume = historical_data["volume"]
    vwap = amount / volume
    maker_volume = volume - taker_volume
    ret = np.log(close) - np.log(close.shift(1))
    factor = Op.quantile_position(vwap, 90)     # quantile 90
    direction = 1
    opposite = 1
    if direction:
        factor = factor * np.sign(ret)
    if opposite:
        factor = factor * -1
    return factor.shift(1)


if __name__ == '__main__':
    trade_uni = get_trade_uni()
    fea_lst = ["close", "close_BTC", "high", "low", "open", "quote_volume", "taker_base_volume", "taker_quote_volume",
               "trades", "volume"]
    freq = "1440min"
    historical_data = get_historical_data(freq, fea_lst)
    factor = generate_factor(historical_data)
