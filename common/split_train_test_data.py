import pandas as pd
import math

train_data_ratio = 0.8


def get_train_data(is_train, df_train):
    train_data_size = math.ceil(1460 * train_data_ratio)
    if is_train:
        x = df_train.iloc[:train_data_size, :(df_train.shape[1]-2)]
        y = df_train.SalePrice[:train_data_size]
    else:
        x = df_train.iloc[train_data_size:, :(df_train.shape[1]-2)]
        y = df_train.SalePrice[train_data_size:]
    return x, y