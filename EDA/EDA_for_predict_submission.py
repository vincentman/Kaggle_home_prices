import pandas as pd

pd_csv = pd.read_csv('../test.csv')

# 刪除 BsmtFinSF1, BsmtFinSF2 欄位，因為上一步已經把 BsmtFinType1, BsmtFinType2 欄位也刪了
# 刪除 Exterior1st, Exterior2nd, BsmtUnfSF 欄位，因為它與 SalePrice 似乎無關
pd_csv = pd_csv.drop(['BsmtFinSF1', 'BsmtFinSF2', 'Exterior1st', 'Exterior2nd'], axis=1)

# print(pd_csv.loc[pd_csv['TotalBsmtSF'].isnull()])
# 將缺值的 TotalBsmtSF 以其平均數取代
pd_csv['TotalBsmtSF'] = pd_csv['TotalBsmtSF'].fillna(pd_csv['TotalBsmtSF'].mean())
# print(pd_csv['TotalBsmtSF'].isnull().sum())

print('Sample whose KitchenQual is null, KitchenAbvGr = ', pd_csv.loc[pd_csv['KitchenQual'].isnull()]['KitchenAbvGr'].values)
# 找出 KitchenAbvGr 值為1的樣本，並列出它們的 KitchenQual
pd_csv.loc[pd_csv['KitchenAbvGr']==1]['KitchenQual']
# 將缺值的 KitchenQual 以 'TA' 取代
pd_csv['KitchenQual'] = pd_csv['KitchenQual'].fillna('TA')
# print(pd_csv['KitchenQual'].isnull().sum())

# print('Sample whose GarageArea is null, GarageType = ', pd_csv.loc[pd_csv['GarageArea'].isnull()]['GarageType'].values)
# 將缺值的 GarageArea 以 GarageType 為 'Detchd' 的樣本，其 GarageArea 的平均數取代
# pd_csv['GarageArea'] = pd_csv['GarageArea'].fillna(int(pd_csv[pd_csv['GarageType']=='Detchd']['GarageArea'].mean()))
# 將缺值的 GarageArea 以其平均數取代
pd_csv['GarageArea'] = pd_csv['GarageArea'].fillna(int(pd_csv['GarageArea'].mean()))
# print(pd_csv['GarageArea'].isnull().sum())

print('Sample whose SaleType is null, SaleCondition = ', pd_csv.loc[pd_csv['SaleType'].isnull()]['SaleCondition'].values)
# 找出 SaleCondition 值為 'Normal' 的樣本，並列出它們的 SaleType
pd_csv.loc[pd_csv['SaleCondition']=='Normal']['SaleType']
# 將缺值的 SaleType 以 'WD' 取代
pd_csv['SaleType'] = pd_csv['SaleType'].fillna('WD')
# print(pd_csv['SaleType'].isnull().sum())

print('Sample whose GarageCars is null, GarageArea = ', pd_csv.loc[pd_csv['GarageCars'].isnull()]['GarageArea'].values)
# 將缺值的 GarageCars 以 2 取代
pd_csv['GarageCars'] = pd_csv['GarageCars'].fillna(2)
# print(pd_csv['GarageCars'].isnull().sum())

