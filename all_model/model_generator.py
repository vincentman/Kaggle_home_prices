from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np

param_alpha_ridge = (np.arange(0.25, 6, 0.25))
# param_alpha_ridge = [.0001, .0003, .0005, .0007, .0009,
#                      .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 60, 70, 80]
model_Ridge = {"model_name": 'Ridge', 'model': Ridge(), 'param_grid': {'alpha': param_alpha_ridge}}

model_ElasticNet = {"model_name": 'ElasticNet', 'model': ElasticNet(),
                    'param_grid': {'alpha': np.arange(1e-4, 1e-3, 1e-4),
                                   'l1_ratio': np.arange(0.1, 1.0, 0.1),
                                   'max_iter': [100000]}
                    }

model_GradientBoostingRegressor = {"model_name": 'GradientBoostingRegressor', 'model': GradientBoostingRegressor(),
                                   'param_grid': {'n_estimators': [150, 250, 350],
                                                  'max_depth': [1, 2, 3],
                                                  'min_samples_split': [5, 6, 7]}}

model_SVR = {"model_name": 'SVR', 'model': SVR(), 'param_grid': {'C': np.arange(1, 21, 2),
                                                                 'kernel': ['poly', 'rbf', 'sigmoid'],
                                                                 'gamma': ['auto']}}

# param_grid_rfr = {'n_estimators': [100, 500, 1000]}
param_grid_rfr = {'n_estimators': [100, 150, 200],
                  'max_features': [25, 50, 75],
                  'min_samples_split': [2, 4, 6]}
model_RandomForestRegressor = {"model_name": 'RandomForestRegressor', 'model': RandomForestRegressor(),
                               'param_grid': param_grid_rfr}


def get_model():
    for i in [model_Ridge]:
        # for i in [model_Ridge, model_ElasticNet, model_GradientBoostingRegressor, model_SVR, model_RandomForestRegressor]:
        yield i
