import numpy as np
from sklearn.externals import joblib
from common import process_data_from_Jack
import pandas as pd

processData = process_data_from_Jack.ProcessData()

processData.generate_clean_data()

x_test = processData.get_test_data()

clf = joblib.load('elastic_net_dump.pkl')

pd.DataFrame({"Id": np.arange(1461, 2920), "SalePrice": np.exp(clf.predict(x_test))}).to_csv(
    'submission.csv', header=True, index=False)