from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
import numpy as np

model_Ridge = {"model_name": 'Ridge', 'model': Ridge(), 'param_grid': {'alpha': (np.arange(0.25, 6, 0.25))}}
model_ElasticNet = {"model_name": 'ElasticNet', 'model': ElasticNet(),
                    'param_grid': {'alpha': np.arange(1e-4, 1e-3, 1e-4),
                                   'l1_ratio': np.arange(0.1, 1.0, 0.1),
                                   'max_iter': [100000]}
                    }


def get_model():
    for i in [model_Ridge]:
    # for i in [model_Ridge, model_ElasticNet]:
        yield i
