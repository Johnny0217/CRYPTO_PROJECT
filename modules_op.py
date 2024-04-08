import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from modules_importation import *


class Op:
    @staticmethod
    def efficient_ratio(feature, windows):
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


if __name__ == '__main__':
    np.random.seed(30)
    df1 = pd.DataFrame(np.random.randint(0, 100, (10, 5)))
    er_ratio = op.efficient_ratio(df1, 5)
    print("debug point here")
