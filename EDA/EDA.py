import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from common import process_data

csv_df = pd.read_csv('../train.csv')

print(
    'Before processing missing value, sample count =>\n{}'.format(process_data.get_missing_value_sample_count(csv_df)))
print(
    'Before processing missing value, sample proportion =>\n{}'.format(
        process_data.get_missing_value_sample_proportion(csv_df)))

print('Sample whose FireplaceQu is null, SalePrice = ',
      csv_df.loc[csv_df['FireplaceQu'].isnull()]['SalePrice'].values)
