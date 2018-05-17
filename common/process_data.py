import numpy as np
import pandas as pd


def get_clean_data(df_train):
    # 查看各欄位 missing value 的數量
    null_count_df = pd.DataFrame(df_train.isnull().sum())
    null_count_df.set_axis(['null_count'], axis=1, inplace=True)
    null_index = null_count_df[null_count_df['null_count'] == 0].index
    print_null_count_df = 'Before processing missing value, sample count =>\n{}'.format(null_count_df.drop(null_index))
    print(print_null_count_df)

    # 補缺值：將 FireplaceQu 缺值的內容補成 'None'
    df_train['FireplaceQu'].fillna('None', inplace=True)

    # 補缺值：將 LotFrontage 缺值的內容補成 0
    df_train['LotFrontage'].fillna(0, inplace=True)

    # 補缺值：將 LotFrontage 缺值的內容用「MasVnrType為BrkFace且Foundation為PConc」的樣本其 LotFrontage 的中位數去補
    mask = (df_train['MasVnrArea'].isnull())
    df_train.loc[mask, 'MasVnrArea'] = \
    df_train[(df_train['MasVnrType'] == 'BrkFace') & (df_train['Foundation'] == 'PConc')][
        'MasVnrArea'].median()

    # 補缺值：將 PoolQC 缺值的內容補成 'None'
    df_train['PoolQC'].fillna('None', inplace=True)

    # 補缺值：將 Fence 缺值的內容補成 'None'
    df_train['Fence'].fillna('None', inplace=True)

    # 刪除缺值樣本數超過1個的欄位
    df_train = df_train.drop((null_count_df[null_count_df['null_count'] > 1]).index, 1)

    # 刪除Electrical欄位缺值的樣本(僅1個樣本)
    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

    # 查看各欄位 missing value 的數量
    null_count_df = pd.DataFrame(df_train.isnull().sum())
    null_count_df.set_axis(['null_count'], axis=1, inplace=True)
    null_index = null_count_df[null_count_df['null_count'] == 0].index
    print_null_count_df = 'After processing missing value, sample count =>\n'.format(null_count_df)
    print(print_null_count_df)

    # 查看各欄位 missing value 的比例
    null_mean_df = pd.DataFrame(df_train.isnull().mean())
    null_mean_df.set_axis(['null_mean'], axis=1, inplace=True)
    null_index = null_mean_df[null_mean_df['null_mean'] == 0.].index
    null_mean_df = null_mean_df.drop(null_index).sort_values(by='null_mean', ascending=False)
    print_null_mean_df = 'After processing missing value, sample proportion =>\n'.format(null_mean_df)
    print(print_null_mean_df)

    # 刪除離群的 GrLivArea 值很高的數據
    ids = df_train.sort_values(by='GrLivArea', ascending=False)[:2]['Id']
    df_train = df_train.drop(ids.index)

    # 將 SalePrice 做對數變換
    df_train['SalePrice'] = np.log(df_train['SalePrice'])
    print('After log transformation, SalePrice skewness is ', df_train['SalePrice'].skew())

    # 將 GrLivArea 做對數變換
    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
    print('After log transformation, GrLivArea skewness is ', df_train['GrLivArea'].skew())

    # 新增'HasBsmt'欄位：表示是否有地下室
    df_train['HasBsmt'] = 0
    df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

    # 將 TotalBsmtSF 做對數變換
    # df_train['TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
    # print('After log transformation, TotalBsmtSF skewness is ', df_train['TotalBsmtSF'].skew())

    return df_train
