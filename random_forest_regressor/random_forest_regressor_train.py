from common import split_train_test_data
from common import process_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.externals import joblib
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

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
# RandomForestRegressor 無法使用 GridSearch
clf = RandomForestRegressor(n_estimators=1000,
                            criterion='mse',
                            random_state=1,
                            # max_features='sqrt',
                            # max_depth=4,
                            n_jobs=-1)
# clf = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                    max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=15, min_samples_split=10,
#                                    loss='ls', random_state =5)
clf.fit(x_train, y_train)
end = time.time()
elapsed_train_time = 'RandomForestRegressor, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                                            int((end - start) % 60))
print(elapsed_train_time)
y_train_pred = clf.predict(x_train)

rmse_print = 'RMSE train: %.3f' % (np.sqrt(mean_squared_error(y_train, y_train_pred)))
print(rmse_print)
r_square_print = 'R^2 train: %.3f' % (r2_score(y_train, y_train_pred))
# r_square_print = 'R^2 train: %.3f' % (clf.score(x_train, y_train))
print(r_square_print)

joblib.dump(clf, 'rf_regressor_dump.pkl')

with open('rf_regressor_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write(rmse_print + '\n')
    file.write(r_square_print + '\n')
