import numba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from modules_importation import *


class Op:
    @staticmethod
    def quantile_position(feature, windows):
        @numba.jit
        def historical_quantile(arr):
            length = len(arr)
            current_number = arr[-1]
            position = np.nansum(arr <= current_number)
            quantile = position / length
            if length > F_ZERO:
                return quantile
            else:
                return np.nan

        df = feature.copy()
        matrix = df.values
        factor_arr = np.full(matrix.shape, np.nan)
        for i in range(matrix.shape[1]):  # loop product
            product_array = matrix[:, i]
            for j in range(windows - 1, len(product_array), 1):
                start = j - windows + 1
                end = j
                arr = product_array[start:end + 1]
                value = historical_quantile(arr)
                factor_arr[end, i] = value
        factor = pd.DataFrame(factor_arr, index=feature.index, columns=feature.columns)
        return factor

    @staticmethod
    def efficient_ratio(feature, windows):
        @numba.jit
        def er(arr):
            numerator = abs(arr[-1] - arr[0])
            denominator = np.nansum(np.abs(np.diff(arr)))
            if denominator > F_ZERO:
                return numerator / denominator
            else:
                return np.nan

        df = feature.copy()
        matrix = df.values
        factor_arr = np.full(matrix.shape, np.nan)
        for i in range(matrix.shape[1]):  # loop product
            product_array = matrix[:, i]
            for j in range(windows - 1, len(product_array), 1):
                start = j - windows + 1
                end = j
                arr = product_array[start:end + 1]
                value = er(arr)
                factor_arr[end, i] = value
        factor = pd.DataFrame(factor_arr, index=feature.index, columns=feature.columns)
        return factor

    @staticmethod
    def ts_regression(feature1, feature2, windows, attribute):
        '''
        :param feature1: x
        :param feature2: y
        :param windows: rolling windows
        :param attribute: beta / intercept
        :return: beta / intercept
        '''

        @numba.jit()
        def ts_regress(x_, y_, attribute):
            if len(x_) < 1:
                return np.nan
            if len(y_) < 1:
                return np.nan
            x = np.where(np.isnan(y_), np.nan, x_)
            y = np.where(np.isnan(x_), np.nan, y_)
            mean_x = np.nanmean(x)
            mean_y = np.nanmean(y)
            numerator = np.nansum((x - mean_x) * (y - mean_y))
            denominator = np.nansum((x - mean_x) ** 2)
            beta_1 = numerator / denominator if denominator > F_ZERO else np.nan
            residual = mean_y - beta_1 * mean_x
            if attribute == "beta":
                return beta_1
            else:
                return residual

        x_matrix = feature1.values
        y_matrix = feature2.values
        factor_arr = np.full(x_matrix.shape, np.nan)
        for i in range(x_matrix.shape[1]):  # product
            x_array = x_matrix[:, i]
            y_array = y_matrix[:, i]
            for j in range(windows - 1, len(x_array), 1):  # rolling window
                start = j - windows + 1
                end = j
                x = x_array[start:end + 1]
                y = y_array[start:end + 1]
                value = ts_regress(x, y, attribute)
                factor_arr[end, i] = value
        factor = pd.DataFrame(factor_arr, index=feature1.index, columns=feature1.columns)
        return factor


if __name__ == '__main__':
    np.random.seed(30)
    df1 = pd.DataFrame(np.random.randint(0, 100, (10, 5)))
    df2 = pd.DataFrame(np.random.randint(0, 100, (10, 5)))
    # er_ratio = Op.efficient_ratio(df1, 7)  # op efficient ratio
    regress = Op.ts_regression(df1, df2, 7, "beta")
    print("debug point here")
