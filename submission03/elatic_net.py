from sklearn.linear_model import ElasticNetCV
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.externals import joblib
from common import process_data_from_Jack

processData = process_data_from_Jack.ProcessData()

processData.generate_clean_data()

x_train, y_train = processData.get_training_data()

start = time.time()

clf = ElasticNetCV(alphas=[0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5,
                   n_jobs=-1, random_state=3)
clf.fit(x_train, y_train)
end = time.time()
elapsed_train_time = 'ElasticNet, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                                 int((end - start) % 60))
print(elapsed_train_time)
y_train_pred = clf.predict(x_train)

rmse_print = 'ElasticNet, RMSE train: %.3f' % (np.sqrt(mean_squared_error(y_train, y_train_pred)))
print(rmse_print)
r_square_print = 'ElasticNet, R^2 train: %.3f' % (r2_score(y_train, y_train_pred))
# r_square_print = 'ElasticNet, R^2 train: %.3f' % (clf.score(x_train, y_train))
print('ElasticNetCV, alpha = ', clf.alpha_)
print('ElasticNetCV, l1_ratio = ', clf.l1_ratio_)

joblib.dump(clf, 'elastic_net_dump.pkl')

with open('elastic_net_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write(rmse_print + '\n')
    file.write(r_square_print + '\n')


# validation ###########
clf = joblib.load('elastic_net_dump.pkl')
x_test, y_test = processData.get_validation_data()
y_test_pred = clf.predict(x_test)
rmse_print = 'ElasticNetCV, test RMSE: %.3f' % (np.sqrt(mean_squared_error(y_test, y_test_pred)))
print(rmse_print)
r_square_print = 'ElasticNetCV, test R^2: %.3f' % (r2_score(y_test, y_test_pred))
print(r_square_print)
with open('elastic_net_predict_info.txt', 'w') as file:
    file.write(rmse_print + '\n')
    file.write(r_square_print + '\n')



