from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

model_Ridge = {"model_name": 'Ridge', 'model': Ridge(), 'param_grid': {'alpha': (np.arange(0.25, 6, 0.25))}}
model_ElasticNet = {"model_name": 'ElasticNet', 'model': ElasticNet(),
                    'param_grid': {'alpha': np.arange(1e-4, 1e-3, 1e-4),
                                   'l1_ratio': np.arange(0.1, 1.0, 0.1),
                                   'max_iter': [100000]}
                    }
model_GradientBoostingRegressor = {"model_name": 'GradientBoostingRegressor', 'model': GradientBoostingRegressor(),
                                   'param_grid': {'n_estimators': [150, 250, 350],
                                                   'max_depth': [1, 2, 3],
                                                   'min_samples_split': [5, 6, 7]}}


def get_model():
    for i in [model_Ridge]:
    # for i in [model_Ridge, model_ElasticNet, model_GradientBoostingRegressor]:
        yield i
