from sklearn.linear_model import ElasticNetCV
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.externals import joblib
from elastic_net_from_Jack_EDA import process_data
import pandas as pd

processData = process_data.ProcessData()

processData.generate_clean_data()

x_test = processData.get_test_data()

clf = joblib.load('elastic_net_dump.pkl')

pd.DataFrame({"Id": np.arange(1461, 2920), "SalePrice": np.exp(clf.predict(x_test))}).to_csv(
    'submission.csv', header=True, index=False)