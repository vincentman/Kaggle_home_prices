import pandas as pd
from sklearn.linear_model import ElasticNetCV
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.externals import joblib
from common import process_data_from_Jack
from common.process_data_from_Jack import ProcessData
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from all_model import model_generator

processData = process_data_from_Jack.ProcessData()

processData.feature_engineering()

x_train, y_train = processData.get_training_data()

# places to store optimal models and scores
best_model_instances = dict()
scores_on_models = pd.DataFrame(columns=['cv_mean', 'cv_std', 'rmse', 'rsquare'])

# no. k-fold splits
splits = 5
# no. k-fold iterations
repeats = 5

model_gen = model_generator.get_model()
while True:
    try:
        model = model_gen.__next__()
    except StopIteration:
        break
    model_name = model['model_name']
    model_instance = model['model']
    model_param_grid = model['param_grid']
    best_model_instances[model_name], score, grid_results = processData.train_model(
        model_instance,
        param_grid=model_param_grid, X=x_train,
        y=y_train,
        splits=splits, repeats=repeats)
    score.name = model_name
    scores_on_models = scores_on_models.append(score)

print('Final score ==========>\n', scores_on_models.sort_values(by='rmse'))
scores_on_models.to_csv(
    'model_train_score.csv', header=True, index=True)
