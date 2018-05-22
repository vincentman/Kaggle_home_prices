from common import split_train_test_data
from common import process_data
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.externals import joblib
import numpy as np

csv_df = pd.read_csv('../test.csv')

x_test = process_data.get_clean_data(csv_df, True)

print('null sum max=======', x_test.isnull().sum().max())
print('After clean, x_test.shape: ', x_test.shape)
print('After clean, x_test.columns => \n', x_test.columns.values)

# 將類別變量轉換為虛擬變量(one-hot encoding)
categorical = [var for var in x_test.columns if x_test[var].dtype == 'O']
for col in categorical:
    x_test[col] = x_test[col].astype('category').cat.codes
print('After clean, x_test(one-hot encoding).shape: ', x_test.shape)

elastic_net_clf = joblib.load('elastic_net_dump.pkl')

pd.DataFrame({"Id": np.arange(1461, 2920), "SalePrice": elastic_net_clf.predict(x_test)}).to_csv(
    'submission.csv', header=True, index=False)