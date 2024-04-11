import importlib
import os.path

from modules_select_coin_backtest_mp import *
from utils import *
from modules_importation import *


def run_factor_from_archive_file(file_name, function_name, *args, **kwargs):
    module = importlib.import_module(f'factors.{file_name}')
    func = getattr(module, function_name)
    return func(*args, **kwargs)


def save_factor_results(factor_name, factor_attribute_dict):
    # factor_dict -> factor attribute dict
    lookback_days = int(factor_attribute_dict[factor_name]["params"].split("_")[0])
    holding_days = int(factor_attribute_dict[factor_name]["params"].split("_")[1])
    threshold = float(factor_attribute_dict[factor_name]["params"].split("_")[2])
    exec_mode = factor_attribute_dict[factor_name]["exec_mode"]
    factor = run_factor_from_archive_file(f"{factor_name}", "generate_factor", historical_data)
    coef_vol = adj_vol_coef(ret, STD_WINDOW, TARGET_VOL)
    volatility_adjust = False
    pos = get_pos(factor, lookback_days, holding_days, threshold, coef_vol, volatility_adjust, exec_mode)
    basecode_return = pos * ret
    pnl = basecode_return.sum(axis=1).cumsum()
    save_path = os.path.join(PROJECT_PATH, "optimization", exec_mode, factor_name)
    check_path(save_path)
    pos_save_path = os.path.join(save_path, "pos.csv")
    basecode_return_save_path = os.path.join(save_path, "basecode_return.csv")
    pnl_save_path = os.path.join(save_path, "pnl.csv")
    print(f"{log_info()} {factor_name} [pos] [basecode_return] [pnl] results saved")
    pos.to_csv(pos_save_path)
    basecode_return.to_csv(basecode_return_save_path)
    pnl.to_csv(pnl_save_path)
    return


if __name__ == '__main__':
    trade_uni = get_trade_uni()
    fea_lst = ["close", "close_BTC", "high", "low", "open", "quote_volume", "taker_base_volume", "taker_quote_volume",
               "trades", "volume"]
    freq = "1440min"
    historical_data = get_historical_data(freq, fea_lst)
    ret = np.log(historical_data["close"]) - np.log(historical_data["close"].shift(1))
    # key dict
    factor_attribute_dict = {
        "factor_er_ratio": {
            "exec_mode": "longshort",
            "params": "15_3_0.5"},
        "factor_quantile": {
            "exec_mode": "longshort",
            "params": "5_5_0.7"},
        "factor_quantile_high_threshold": {
            "exec_mode": "longshort",
            "params": "8_3_1.3"},
        "factor_er_ratio_high_threshold": {
            "exec_mode": "longshort",
            "params": "15_2_1.2"}
    }
    # save single factors
    # for factor_name in factor_attribute_dict.keys():
    #     save_factor_results(factor_name, factor_attribute_dict)
    save_factor_results("factor_er_ratio_high_threshold", factor_attribute_dict)

    # correlation analysis
    factor_corr_list = ["factor_er_ratio", "factor_quantile", "factor_quantile_high_threshold",
                        "factor_er_ratio_high_threshold"]
