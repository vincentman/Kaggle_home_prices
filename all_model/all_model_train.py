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
train_scores_on_models = pd.DataFrame(columns=['rmse', 'rsquare', 'cv_mean', 'cv_std', 'train_time'])

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
    start = time.time()
    best_model_instances[model_name], (rmse, rsquare, cv_mean, cv_std), grid_results = processData.train_model(
        model_instance,
        param_grid=model_param_grid, X=x_train,
        y=y_train,
        splits=splits, repeats=repeats)
    end = time.time()
    train_time = '%.2fmin' % ((end - start) / 60)
    score = pd.Series(
        {'rmse': rmse, 'rsquare': rsquare, 'cv_mean': cv_mean, 'cv_std': cv_std, 'train_time': train_time})
    score.name = model_name
    train_scores_on_models = train_scores_on_models.append(score)

print('============= Final train score =============\n', train_scores_on_models.sort_values(by='rmse'),
      '\n=====================================')
train_scores_on_models.to_csv(
    'model_train_score.csv', header=True, index=True)

# validation ###########
validation_scores_on_models = pd.DataFrame(columns=['rmse', 'rsquare'])
for model_name, model_instance in best_model_instances.items():
    x_test, y_test = processData.get_validation_data()
    y_test_pred = model_instance.predict(x_test)
    rsquare = '%.5f' % r2_score(y_test, y_test_pred)
    # print('validation, R^2=', rsquare)
    rmse = '%.5f' % ProcessData.rmse(y_test, y_test_pred)
    # print('validation, RMSE=', rmse)
    score = pd.Series({'rmse': rmse, 'rsquare': rsquare})
    score.name = model_name
    validation_scores_on_models = validation_scores_on_models.append(score)

print('========== Final validation score ==========\n', validation_scores_on_models.sort_values(by='rmse'),
      '\n======================================')
validation_scores_on_models.to_csv(
    'model_validation_score.csv', header=True, index=True)
