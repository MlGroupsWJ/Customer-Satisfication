# -*- coding:UTF-8 -*-
import xgboost as xgb
from EDA.EDA5 import train_data, test_data, IDlist
from Common.ModelCommon import *
import os

x_train = train_data.iloc[:, :-1]
y_train = train_data.TARGET

# 增加各行0的统计
# x_train['n0'] = (x_train == 0).sum(axis=1)
# train_data['n0'] = x_train['n0']
# test_data['n0'] = (test_data == 0).sum(axis=1)
# train_data['n0'] = test_data['n0']

params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'gbtree'
params['eval_metric'] = 'auc'
params['eta'] = 0.02
params['max_depth'] = 3
params['min_child_weight'] = 5
params['subsample'] = 0.7
params['colsample_bytree'] = 0.5
params['early_stopping_rounds'] = 30

d_train = xgb.DMatrix(x_train, label=y_train)
watchlist = [(d_train, 'train')]

clf = xgb.train(params, d_train, 600, watchlist)

d_test = xgb.DMatrix(test_data)
y_pred = clf.predict(d_test)
submission = pd.DataFrame({"ID": IDlist, "TARGET": y_pred})
submission.to_csv("../Result/EDA5_myXGB.csv", index=False)