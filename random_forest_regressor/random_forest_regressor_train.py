from common import split_train_test_data
from common import process_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.externals import joblib
import time
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

pd_csv = pd.read_csv('../train.csv')

df_train = process_data.get_clean_data(pd_csv)

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
# clf = RandomForestRegressor(n_estimators=1000,
#                             criterion='mse',
#                             random_state=1,
#                             n_jobs=-1)
clf = ElasticNet(alpha=0.0005, l1_ratio=.5, random_state=3)
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
