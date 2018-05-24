from common import split_train_test_data
from common import process_data
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.externals import joblib
import time
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV

csv_df = pd.read_csv('../train.csv')

print(
    'Before processing missing value, sample count =>\n{}'.format(process_data.get_missing_value_sample_count(csv_df)))
print(
    'Before processing missing value, sample proportion =>\n{}'.format(
        process_data.get_missing_value_sample_proportion(csv_df)))

df_train = process_data.get_clean_data(csv_df)

# 將 SalePrice 做對數變換
df_train['SalePrice'] = np.log(df_train['SalePrice'])
print('After log transformation, SalePrice skewness is ', df_train['SalePrice'].skew())

# 刪除Electrical欄位缺值的樣本(僅1個樣本)
print('Sample(Id={}) is dropped due to Electrical is null'.format(
    df_train.loc[df_train['Electrical'].isnull()]['Id'].values))
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

# 刪除離群的 GrLivArea 值很高的數據
ids = df_train.sort_values(by='GrLivArea', ascending=False)[:2]['Id']
df_train = df_train.drop(ids.index)

print(
    'After processing missing value, sample count =>\n{}'.format(process_data.get_missing_value_sample_count(df_train)))
print(
    'After processing missing value, sample proportion =>\n{}'.format(
        process_data.get_missing_value_sample_proportion(df_train)))

x_train, y_train = split_train_test_data.get_splitted_data(True, df_train)

print('After clean, x_train.shape: ', x_train.shape)
print('After clean, x_train.columns => \n', x_train.columns.values)
print('After clean, y_train.shape: ', y_train.shape)

# 將類別變量轉換為虛擬變量(one-hot encoding)
# x_train = pd.get_dummies(x_train)
# print('x_train(dummy).shape: ', x_train.shape)
# print('x_train(dummy).columns => \n', x_train.columns.values)

categorical = [var for var in x_train.columns if x_train[var].dtype == 'O']
for col in categorical:
    x_train[col] = x_train[col].astype('category').cat.codes
print('After clean, x_train(one-hot encoding).shape: ', x_train.shape)
# print('x_train(one-hot encoding).columns => \n', x_train.columns.values)

start = time.time()

clf = ElasticNetCV(alphas=[0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5,
                   n_jobs=-1, random_state=3)
clf.fit(x_train, y_train)
end = time.time()
elapsed_train_time = 'ElasticNet, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                                 int((end - start) % 60))
print(elapsed_train_time)
y_train_pred = clf.predict(x_train)

rmse_print = 'ElasticNet, RMSE train: %.3f' % (np.sqrt(mean_squared_error(y_train, y_train_pred)))
print(rmse_print)
r_square_print = 'ElasticNet, R^2 train: %.3f' % (r2_score(y_train, y_train_pred))
# r_square_print = 'ElasticNet, R^2 train: %.3f' % (clf.score(x_train, y_train))
print('ElasticNetCV, alpha = ', clf.alpha_)
print('ElasticNetCV, l1_ratio = ', clf.l1_ratio_)

joblib.dump(clf, 'elastic_net_dump.pkl')

with open('elastic_net_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write(rmse_print + '\n')
    file.write(r_square_print + '\n')
