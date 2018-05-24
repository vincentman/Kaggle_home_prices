import pandas as pd
from common import process_data

csv_df = pd.read_csv('../test.csv')

csv_df = process_data.get_clean_data(csv_df)

print(
    'Before processing missing value, sample count =>\n{}'.format(process_data.get_missing_value_sample_count(csv_df)))
print(
    'Before processing missing value, sample proportion =>\n{}'.format(
        process_data.get_missing_value_sample_proportion(csv_df)))

# print(pd_csv.loc[pd_csv['TotalBsmtSF'].isnull()])
# 將缺值的 TotalBsmtSF 以其平均數取代
csv_df['TotalBsmtSF'] = csv_df['TotalBsmtSF'].fillna(csv_df['TotalBsmtSF'].mean())
# print(pd_csv['TotalBsmtSF'].isnull().sum())

print('Sample whose KitchenQual is null, KitchenAbvGr = ',
      csv_df.loc[csv_df['KitchenQual'].isnull()]['KitchenAbvGr'].values)
# 找出 KitchenAbvGr 值為1的樣本，並列出它們的 KitchenQual
csv_df.loc[csv_df['KitchenAbvGr'] == 1]['KitchenQual']
# 將缺值的 KitchenQual 以 'TA' 取代
csv_df['KitchenQual'] = csv_df['KitchenQual'].fillna('TA')
# print(pd_csv['KitchenQual'].isnull().sum())

# print('Sample whose GarageArea is null, GarageType = ', pd_csv.loc[pd_csv['GarageArea'].isnull()]['GarageType'].values)
# 將缺值的 GarageArea 以 GarageType 為 'Detchd' 的樣本，其 GarageArea 的平均數取代
# pd_csv['GarageArea'] = pd_csv['GarageArea'].fillna(int(pd_csv[pd_csv['GarageType']=='Detchd']['GarageArea'].mean()))
# 將缺值的 GarageArea 以其平均數取代
csv_df['GarageArea'] = csv_df['GarageArea'].fillna(int(csv_df['GarageArea'].mean()))
# print(pd_csv['GarageArea'].isnull().sum())

print('Sample whose SaleType is null, SaleCondition = ',
      csv_df.loc[csv_df['SaleType'].isnull()]['SaleCondition'].values)
# 找出 SaleCondition 值為 'Normal' 的樣本，並列出它們的 SaleType
csv_df.loc[csv_df['SaleCondition'] == 'Normal']['SaleType']
# 將缺值的 SaleType 以 'WD' 取代
csv_df['SaleType'] = csv_df['SaleType'].fillna('WD')
# print(pd_csv['SaleType'].isnull().sum())

print('Sample whose GarageCars is null, GarageArea = ', csv_df.loc[csv_df['GarageCars'].isnull()]['GarageArea'].values)
# 將缺值的 GarageCars 以 2 取代
csv_df['GarageCars'] = csv_df['GarageCars'].fillna(2)
# print(pd_csv['GarageCars'].isnull().sum())

# print('Sample whose MSZoning is null, MSSubClass = \n', csv_df.loc[csv_df['MSZoning'].isnull()][['Id', 'MSSubClass']])
# 找出 MSSubClass 值為 20/30/70 的樣本，並列出它們的 MSZoning
csv_df.loc[csv_df['MSSubClass'] == 20]['MSZoning']
csv_df.loc[csv_df['MSSubClass'] == 30]['MSZoning']
csv_df.loc[csv_df['MSSubClass'] == 70]['MSZoning']
# 將缺值的 MSZoning 以 'RL' 取代
# mask1 = (csv_df['MSZoning'].isnull()) & (csv_df['MSSubClass'] == 20)
# csv_df.loc[mask1, "MSZoning"] = csv_df.loc[mask1, "MSZoning"].fillna('RL')
csv_df['MSZoning'] = csv_df['MSZoning'].fillna('RL')
# print(csv_df['MSZoning'].isnull().sum())

# 將缺值的 Utilities 以最常見的值取代
csv_df['Utilities'] = csv_df['Utilities'].fillna(csv_df['Utilities'].value_counts().index[0])

# 將缺值的 Functional 以最常見的值取代
csv_df['Functional'] = csv_df['Functional'].fillna(csv_df['Functional'].value_counts().index[0])

print(
    'After processing missing value, sample count =>\n{}'.format(process_data.get_missing_value_sample_count(csv_df)))
print(
    'After processing missing value, sample proportion =>\n{}'.format(
        process_data.get_missing_value_sample_proportion(csv_df)))
