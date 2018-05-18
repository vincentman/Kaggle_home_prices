import math

train_data_ratio = 0.8


def get_splitted_data(is_train, data):
    train_data_size = math.ceil(1460 * train_data_ratio)
    if is_train:
        y = data.SalePrice[:train_data_size]
        data = data.drop('SalePrice', axis=1)
        x = data.iloc[:train_data_size, :(data.shape[1])]
    else:
        y = data.SalePrice[train_data_size:]
        data = data.drop('SalePrice', axis=1)
        x = data.iloc[train_data_size:, :(data.shape[1])]
    return x, y