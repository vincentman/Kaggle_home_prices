from common import process_data
import pandas as pd
from sklearn.externals import joblib
import numpy as np

df_csv = pd.read_csv('../test.csv')

print(
    'Before processing missing value, sample count =>\n{}'.format(process_data.get_missing_value_sample_count(df_csv)))
print(
    'Before processing missing value, sample proportion =>\n{}'.format(
        process_data.get_missing_value_sample_proportion(df_csv)))

x_test = process_data.get_clean_data(df_csv)

print(
    'After processing missing value, sample count =>\n{}'.format(process_data.get_missing_value_sample_count(x_test)))
print(
    'After processing missing value, sample proportion =>\n{}'.format(
        process_data.get_missing_value_sample_proportion(x_test)))

print('After clean, x_test.isnull().sum().max(): ', x_test.isnull().sum().max())
print('After clean, x_test.shape: ', x_test.shape)
print('After clean, x_test.columns => \n', x_test.columns.values)

# 將類別變量轉換為虛擬變量(one-hot encoding)
categorical = [var for var in x_test.columns if x_test[var].dtype == 'O']
for col in categorical:
    x_test[col] = x_test[col].astype('category').cat.codes
print('After clean, x_test(one-hot encoding).shape: ', x_test.shape)

elastic_net_clf = joblib.load('elastic_net_dump.pkl')

pd.DataFrame({"Id": np.arange(1461, 2920), "SalePrice": np.exp(elastic_net_clf.predict(x_test))}).to_csv(
    'submission.csv', header=True, index=False)