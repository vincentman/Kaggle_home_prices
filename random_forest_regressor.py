from common import split_train_test_data
from common import process_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd

# x, y = load_csv.load_data(True, 'train.csv')
train = pd.read_csv('train.csv')

df_train = process_data.get_clean_data(train)
# df_train = df_train.drop(['SalePrice'], axis=1)

x_train, y_train = split_train_test_data.get_train_data(True, df_train)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y_train.shape: ', y_train.shape)

# 將類別變量轉換為虛擬變量(one-hot encoding)
x_train = pd.get_dummies(x_train)

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(x_train, y_train)
y_train_pred = forest.predict(x_train)

print('MSE train: %.3f' % (mean_squared_error(y_train, y_train_pred)))
print('R^2 train: %.3f' % (r2_score(y_train, y_train_pred)))
