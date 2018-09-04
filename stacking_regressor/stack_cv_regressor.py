from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import GridSearchCV
from common import process_data_from_Jack
import pandas as pd
from common.process_data_from_Jack import ProcessData
from sklearn.metrics import make_scorer
from sklearn.externals import joblib
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning, append=True)

processData = process_data_from_Jack.ProcessData()

processData.feature_engineering()

x_train, y_train = processData.get_training_data()

# param_alpha = np.arange(1e-4, 1e-3, 1e-4)
param_alpha = [0.1, 1, 10]

# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)
stregr = StackingCVRegressor(regressors=[Ridge(), Lasso()],
# stregr = StackingCVRegressor(regressors=[Ridge(), Lasso(), RandomForestRegressor(random_state=RANDOM_SEED)],
                             # meta_regressor=ElasticNet(),
                             meta_regressor=ElasticNet(),
                             use_features_in_secondary=True)

# rfr_param_n_estimators = [100]
elastic_net_param_alpha = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
# elastic_net_param_alpha = np.arange(1e-4, 1e-3, 1e-4)
elastic_net_param_l1_ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
# elastic_net_param_l1_ratio = np.arange(0.1, 1.0, 0.1)
# param_grid = {'ridge__alpha': param_alpha, 'lasso__alpha': param_alpha,
#                               'meta-elasticnet__alpha': elastic_net_param_alpha,
#                               'meta-elasticnet__l1_ratio': elastic_net_param_l1_ratio,
#                               'meta-elasticnet__max_iter':[1000]
#                               }
param_grid = {'ridge__alpha': param_alpha, 'lasso__alpha': param_alpha,
              # 'randomforestregressor__n_estimators': rfr_param_n_estimators,
              'meta-elasticnet__alpha': elastic_net_param_alpha,
              'meta-elasticnet__l1_ratio': elastic_net_param_l1_ratio,
              }
gs = GridSearchCV(estimator=stregr,
                  param_grid=param_grid,
                  cv=5,
                  scoring=make_scorer(ProcessData.rmse, greater_is_better=False),
                  refit=True, return_train_score=True, verbose=1)
start = time.time()
gs.fit(x_train.values, y_train.values)
end = time.time()
elapsed_train_time = 'StackingCVRegressor, elapsed training time: {} min, {} sec '.format(
    int((end - start) / 60),
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
rsquare = '%.5f' % r2_score(y_train, y_train_pred)
# print('R^2=', rsquare)
rmse = '%.5f' % ProcessData.rmse(y_train, y_train_pred)
# print('RMSE=', rmse)
cv_mean = '%.5f' % (abs(grid_results.loc[best_idx, 'mean_test_score']))
cv_std = '%.5f' % (grid_results.loc[best_idx, 'std_test_score'])
train_score_info = 'StackingRegressor, train score: RMSE=%s, R^2=%s, cv_mean=%s, cv_std=%s' % (
    rmse, rsquare, cv_mean, cv_std)
print(train_score_info)

joblib.dump(best_model, 'stack_cv_regressor_dump.pkl')

with open('stack_cv_regressor_score_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write('--------------------------------------------\n')
    file.write(repr(best_model) + '\n')
    file.write('--------------------------------------------\n')
    file.write(train_score_info + '\n')

# validation ###########
x_test, y_test = processData.get_validation_data()
clf = joblib.load('stack_cv_regressor_dump.pkl')
y_test_pred = clf.predict(x_test)
validation_score_info = 'StackingRegressor, validation score: RMSE=%.3f, R^2=%.3f' % (
    np.sqrt(mean_squared_error(y_test, y_test_pred)), r2_score(y_test, y_test_pred))
print(validation_score_info)
with open('stack_cv_regressor_score_info.txt', 'a') as file:
    file.write(validation_score_info + '\n')
