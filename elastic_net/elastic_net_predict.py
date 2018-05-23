from common import split_train_test_data
from common import process_data
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.externals import joblib
import numpy as np

df_csv = pd.read_csv('../train.csv')

df_test = process_data.get_clean_data(df_csv)

# 將 SalePrice 做對數變換
df_test['SalePrice'] = np.log(df_test['SalePrice'])
print('After log transformation, SalePrice skewness is ', df_test['SalePrice'].skew())

# 刪除Electrical欄位缺值的樣本(僅1個樣本)
print('Sample(Id={}) is dropped due to Electrical is null'.format(
    df_test.loc[df_test['Electrical'].isnull()]['Id'].values))
df_test = df_test.drop(df_test.loc[df_test['Electrical'].isnull()].index)

# 刪除離群的 GrLivArea 值很高的數據
ids = df_test.sort_values(by='GrLivArea', ascending=False)[:2]['Id']
df_test = df_test.drop(ids.index)

x_test, y_test = split_train_test_data.get_splitted_data(False, df_test)

print('After clean, x_test.shape: ', x_test.shape)
print('After clean, x_test.columns => \n', x_test.columns.values)
print('After clean, y_test.shape: ', y_test.shape)

# 將類別變量轉換為虛擬變量(one-hot encoding)
# x_test = pd.get_dummies(x_test)
# print('x_test(dummy).shape: ', x_test.shape)
# print('x_test(dummy).columns => \n', x_test.columns.values)

categorical = [var for var in x_test.columns if x_test[var].dtype == 'O']
for col in categorical:
    x_test[col] = x_test[col].astype('category').cat.codes
print('After clean, x_test(one-hot encoding).shape: ', x_test.shape)
# print('x_test(one-hot encoding).columns => \n', x_test.columns.values)

clf = joblib.load('elastic_net_dump.pkl')
y_test_pred = clf.predict(x_test)
rmse_print = 'ElasticNetCV, test RMSE: %.3f' % (np.sqrt(mean_squared_error(y_test, y_test_pred)))
print(rmse_print)
r_square_print = 'ElasticNetCV, test R^2: %.3f' % (r2_score(y_test, y_test_pred))
print(r_square_print)
with open('elastic_net_predict_info.txt', 'w') as file:
    file.write(rmse_print + '\n')
    file.write(r_square_print + '\n')