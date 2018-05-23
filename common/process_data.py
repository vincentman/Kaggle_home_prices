import numpy as np
import pandas as pd


# 查看各欄位 missing value 的數量
def get_missing_value_sample_count(df):
    null_count_df = pd.DataFrame(df.isnull().sum())
    null_count_df.set_axis(['null_count'], axis=1, inplace=True)
    null_index = null_count_df[null_count_df['null_count'] == 0].index
    null_count_df = null_count_df.drop(null_index).sort_values(by='null_count', ascending=False)
    return null_count_df


# 查看各欄位 missing value 的比例
def get_missing_value_sample_proportion(df):
    null_mean_df = pd.DataFrame(df.isnull().mean())
    null_mean_df.set_axis(['null_mean'], axis=1, inplace=True)
    null_index = null_mean_df[null_mean_df['null_mean'] == 0.].index
    null_mean_df = null_mean_df.drop(null_index).sort_values(by='null_mean', ascending=False)
    return null_mean_df


def get_clean_data(csv_df):
    # 補缺值：將 FireplaceQu 缺值的內容補成 'None'
    csv_df['FireplaceQu'].fillna('None', inplace=True)

    # 補缺值：將 LotFrontage 缺值的內容補成 0
    csv_df['LotFrontage'].fillna(0, inplace=True)

    # 補缺值：將 LotFrontage 缺值的內容用「MasVnrType為BrkFace且Foundation為PConc」的樣本其 LotFrontage 的中位數去補
    mask = (csv_df['MasVnrArea'].isnull())
    csv_df.loc[mask, 'MasVnrArea'] = \
        csv_df[(csv_df['MasVnrType'] == 'BrkFace') & (csv_df['Foundation'] == 'PConc')][
            'MasVnrArea'].median()

    # 補缺值：將 PoolQC 缺值的內容補成 'None'
    csv_df['PoolQC'].fillna('None', inplace=True)

    # 補缺值：將 Fence 缺值的內容補成 'None'
    csv_df['Fence'].fillna('None', inplace=True)

    # 刪除缺值樣本數超過1個的欄位
    csv_df = csv_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage',
                          'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',
                          'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual',
                          'MasVnrArea', 'MasVnrType'], axis=1)

    # 刪除 BsmtFinSF1, BsmtFinSF2 欄位，因為上一步已經把 BsmtFinType1, BsmtFinType2 欄位也刪了
    # 刪除 Exterior1st, Exterior2nd, BsmtUnfSF 欄位，因為它與 SalePrice 似乎無關
    csv_df = csv_df.drop(['BsmtFinSF1', 'BsmtFinSF2', 'Exterior1st', 'Exterior2nd'], axis=1)

    # 將 GrLivArea 做對數變換
    csv_df['GrLivArea'] = np.log(csv_df['GrLivArea'])
    print('After log transformation, GrLivArea skewness is ', csv_df['GrLivArea'].skew())

    # 新增'HasBsmt'欄位：表示是否有地下室
    # csv_df['HasBsmt'] = 0
    # csv_df.loc[csv_df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

    # 將 TotalBsmtSF 做對數變換
    # df_train['TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
    # print('After log transformation, TotalBsmtSF skewness is ', df_train['TotalBsmtSF'].skew())

    return csv_df
