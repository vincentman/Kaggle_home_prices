import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

train = pd.read_csv('train.csv')
y_train = train.SalePrice

null_count_df = pd.DataFrame(train.isnull().sum())
null_count_df.set_axis(['null_count'], axis=1, inplace=True)
null_index = null_count_df[null_count_df['null_count'] == 0].index
print(null_count_df.drop(null_index))

null_mean_df = pd.DataFrame(train.isnull().mean())
null_mean_df.set_axis(['null_mean'], axis=1, inplace=True)
null_index = null_mean_df[null_mean_df['null_mean'] == 0.].index
print(null_mean_df.drop(null_index))
