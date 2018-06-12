from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import GridSearchCV
from common import process_data_from_Jack
import pandas as pd
from common.process_data_from_Jack import ProcessData

processData = process_data_from_Jack.ProcessData()

processData.feature_engineering()

x_train, y_train = processData.get_training_data()

param_alpha_ridge = (np.arange(0.25, 6, 0.25))
# param_alpha_ridge = [.0001, .0003, .0005, .0007, .0009,
#                      .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 60, 70, 80]
model_Ridge = {"model_name": 'Ridge', 'model': Ridge(), 'param_grid': {'alpha': param_alpha_ridge}}

elasticnet_param_alpha = np.arange(1e-4, 1e-3, 1e-4)
elasticnet_param_l1_ratio = np.arange(0.1, 1.0, 0.1)
elasticnet_param_max_iter = [100000]
model_ElasticNet = {"model_name": 'ElasticNet', 'model': ElasticNet(),
                    'param_grid': {'alpha': elasticnet_param_alpha,
                                   'l1_ratio': elasticnet_param_l1_ratio,
                                   'max_iter': elasticnet_param_max_iter}
                    }

stregr = StackingRegressor(regressors=[Ridge(), ElasticNet()],
                           meta_regressor=ElasticNet())

param_grid = {'ridge__alpha': param_alpha_ridge, 'elasticnet__alpha': elasticnet_param_alpha,
              'elasticnet__l1_ratio': elasticnet_param_l1_ratio, 'elasticnet__max_iter': elasticnet_param_max_iter,
              'meta-elasticnet__alpha': elasticnet_param_alpha,
              'meta-elasticnet__l1_ratio': elasticnet_param_l1_ratio,
              'meta-elasticnet__max_iter': elasticnet_param_max_iter}

gs = GridSearchCV(estimator=stregr,
                  param_grid=param_grid,
                  cv=5,
                  refit=True)
start = time.time()
gs.fit(x_train, y_train)
end = time.time()
elapsed_train_time = 'ElasticNet, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                                 int((end - start) % 60))
print(elapsed_train_time)

best_model = gs.best_estimator_
best_idx = gs.best_index_
grid_results = pd.DataFrame(gs.cv_results_)
y_train_pred = best_model.predict(x_train)

# print stats on model performance
print('--------------------------------------------')
print(best_model)
print('--------------------------------------------')
rsquare = '%.5f' % best_model.score(x_train, y_train)
# print('R^2=', rsquare)
rmse = '%.5f' % ProcessData.rmse(y_train, y_train_pred)
# print('RMSE=', rmse)
cv_mean = '%.5f' % (abs(grid_results.loc[best_idx, 'mean_test_score']))
cv_std = '%.5f' % (grid_results.loc[best_idx, 'std_test_score'])
print('StackingRegressor, RMSE=%s, R^2=%s' %(rmse, rsquare))
