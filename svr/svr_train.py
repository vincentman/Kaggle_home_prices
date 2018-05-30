from common import split_train_test_data
from common import process_data
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.externals import joblib
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold

pd_csv = pd.read_csv('../train.csv')

df_train = process_data.get_clean_data(pd_csv)

# 將 SalePrice 做對數變換
df_train['SalePrice'] = np.log(df_train['SalePrice'])
print('After log transformation, SalePrice skewness is ', df_train['SalePrice'].skew())

# 刪除Electrical欄位缺值的樣本(僅1個樣本)
print('Sample(Id={}) is dropped due to Electrical is null'.format(
    df_train.loc[df_train['Electrical'].isnull()]['Id'].values))
csv_df = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

x_train, y_train = split_train_test_data.get_splitted_data(True, df_train)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y_train.shape: ', y_train.shape)

# 將類別變量轉換為虛擬變量(one-hot encoding)
# x_train = pd.get_dummies(x_train)
# print('x_train(dummy).shape: ', x_train.shape)
# print('x_train(dummy).columns => \n', x_train.columns.values)

categorical = [var for var in x_train.columns if x_train[var].dtype == 'O']
for col in categorical:
    x_train[col] = x_train[col].astype('category').cat.codes
print('x_train(one-hot encoding).shape: ', x_train.shape)
# print('x_train(one-hot encoding).columns => \n', x_train.columns.values)

start = time.time()
clf = SVR(verbose=True)
kfold = StratifiedKFold(n_splits=10)
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_range = np.arange(1, 21, 2)
param_grid = {'C': param_range,
              'kernel': ['poly', 'rbf', 'sigmoid'],
              'gamma': ['auto']}
              # 'gamma': param_range}
gs = GridSearchCV(estimator=clf,
                  param_grid=param_grid,
                  scoring='neg_mean_squared_error',
                  cv=5)
gs.fit(x_train, y_train)
best_score = 'SVR at GridSearch, best score: {}'.format(gs.best_score_)
print('\n', best_score)
best_param = 'SVR at GridSearch, train best param: {}'.format(gs.best_params_)
print(best_param)
end = time.time()
elapsed_train_time = 'SVR, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                          int((end - start) % 60))
print(elapsed_train_time)