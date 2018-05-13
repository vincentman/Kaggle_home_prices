import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

df_train = pd.read_csv('train.csv')

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# 刪除缺值樣本數超過1個的欄位
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)

# 刪除Electrical欄位缺值的樣本
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

# 檢查是否還有缺值的欄位
print('missing value max count: ', df_train.isnull().sum().max())

# 刪除離群的 GrLivArea 值很高的數據
ids = df_train.sort_values(by='GrLivArea', ascending=False)[:2]['Id']
df_train = df_train.drop(ids.index)

# 將 SalePrice 做對數變換
df_train['SalePrice'] = np.log(df_train['SalePrice'])

# 將 GrLivArea 做對數變換
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

# 新增'HasBsmt'欄位：表示是否有地下室
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

# 將 TotalBsmtSF 做對數變換
df_train['TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

# 將類別變量轉換為虛擬變量(one-hot encoding)
df_train = pd.get_dummies(df_train)

print('columns: \n', df_train.columns.values)
