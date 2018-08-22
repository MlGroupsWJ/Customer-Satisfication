# -*- coding:UTF-8 -*-
import xgboost as xgb
from EDA.EDA3 import train_data, test_data, IDlist
from Common.EDACommon import *
from Common.ModelCommon import *
import os

params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'gbtree'
params['eval_metric'] = 'auc'
params['eta'] = 0.01
params['max_depth'] = 3
params['min_child_weight'] = 5
params['subsample'] = 0.7
params['colsample_bytree'] = 0.5
params['verbose'] = 2
params['maximise'] = False

ratelist = np.arange(0.5, 20, 0.5).tolist()
for rate in ratelist:
    train = underSampling(train_data, rate)
    x_train = train.iloc[:, :-1]
    y_train = train.TARGET
    d_train = xgb.DMatrix(x_train, label=y_train)
    watchlist = [(d_train, 'train')]
    clf = xgb.train(params, d_train, 570, watchlist)
    d_test = xgb.DMatrix(test_data)
    y_pred = clf.predict(d_test)
    submission = pd.DataFrame({"ID": IDlist, "TARGET": y_pred})
    filename = "../Result/EDA3_myXGB_unsample_%d.csv" % ratelist.index(rate)
    submission.to_csv(filename, index=False)
    os.popen(
        'kaggle competitions submit -c santander-customer-satisfaction -f %s -m "xgbparam2，num_boost_round=570,rate:%f"'
        % (filename, rate))

