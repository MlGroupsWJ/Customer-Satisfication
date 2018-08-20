# -*- coding:UTF-8 -*-
from Common.ModelCommon import *
from EDA.EDA3 import train_data, test_data, IDlist
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'gbtree'
params['eval_metric'] = 'auc'

param_search = {
    # 'eta': np.arange(0.01, 0.2, 0.02).tolist(),
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2),
    'subsample': np.arange(0.5, 1, 0.1).tolist(),
    'colsample_bytree': np.arange(0.5, 1, 0.1).tolist()
}

x_train = train_data.iloc[:, :-1]
y_train = train_data.TARGET

xgbc = XGBClassifier(**params)
gridsearch = GridSearchCV(estimator=xgbc, param_grid=param_search, scoring='roc_auc', cv=5)
gridsearch.fit(x_train, y_train)
print(gridsearch.best_params_, gridsearch.best_score_)
# 根据以上参数的grid search结果如下
# {'colsample_bytree': 0.5, 'max_depth': 3, 'min_child_weight': 5, 'subsample': 0.7} 0.8404591709763296