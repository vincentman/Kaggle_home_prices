import pandas as pd

train = pd.read_csv('../train.csv')
y_train = train.SalePrice

print('continuous type columns =>\n', train.describe().columns.values)
print('discrete type columns =>\n', train.describe(include=['O']).columns.values)

# 補缺值：將 FireplaceQu 缺值的內容補成 'None'
train['FireplaceQu'].fillna('None', inplace=True)

# 補缺值：將 LotFrontage 缺值的內容補成 0
train['LotFrontage'].fillna(0, inplace=True)

# 補缺值：將 LotFrontage 缺值的內容用「MasVnrType為BrkFace且Foundation為PConc」的樣本其 LotFrontage 的中位數去補
mask = (train['MasVnrArea'].isnull())
train.loc[mask, 'MasVnrArea'] = train[(train['MasVnrType'] == 'BrkFace') & (train['Foundation'] == 'PConc')][
    'MasVnrArea'].median()

# 補缺值：將 PoolQC 缺值的內容補成 'None'
train['PoolQC'].fillna('None', inplace=True)

# 補缺值：將 Fence 缺值的內容補成 'None'
train['Fence'].fillna('None', inplace=True)

# 查看各欄位 missing value 的數量
null_count_df = pd.DataFrame(train.isnull().sum())
null_count_df.set_axis(['null_count'], axis=1, inplace=True)
null_index = null_count_df[null_count_df['null_count'] == 0].index
print(null_count_df.drop(null_index))

# 查看各欄位 missing value 的比例
null_mean_df = pd.DataFrame(train.isnull().mean())
null_mean_df.set_axis(['null_mean'], axis=1, inplace=True)
null_index = null_mean_df[null_mean_df['null_mean'] == 0.].index
print(null_mean_df.drop(null_index).sort_values(by='null_mean', ascending=False))

