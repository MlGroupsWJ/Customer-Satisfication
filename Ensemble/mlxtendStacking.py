from mlxtend.classifier import StackingClassifier
from xgboost import XGBClassifier
from sklearn import linear_model
from bestScore import *


params1 = {}
params1['objective'] = 'binary:logistic'
params1['booster'] = 'gbtree'
params1['learning_rate'] = 0.02
params1['max_depth'] = 5
params1['subsample'] = 0.6
params1['colsample_bytree'] = 0.5
params1['n_estimators'] = 580

params2 = {}
params2['objective'] = 'binary:logistic'
params2['booster'] = 'gbtree'
params2['learning_rate'] = 0.02
params2['max_depth'] = 5
params2['subsample'] = 0.6
params2['colsample_bytree'] = 0.5
params2['n_estimators'] = 500

params3 = {}
params3['objective'] = 'binary:logistic'
params3['booster'] = 'gbtree'
params3['learning_rate'] = 0.02
params3['max_depth'] = 4
params3['subsample'] = 0.6
params3['colsample_bytree'] = 0.5
params3['n_estimators'] = 600

clf1 = XGBClassifier(**params1)
clf2 = XGBClassifier(**params2)
clf3 = XGBClassifier(**params3)
clfs = [clf1, clf2, clf3]

lrc = linear_model.LogisticRegression(C=0.5, max_iter=300)

x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
mlxcf = StackingClassifier(clfs, lrc, use_probas=True, average_probas=True)
mlxcf.fit(x_train, y_train)
y_pred = mlxcf.predict_proba(test_data)[:, -1]
submission = pd.DataFrame({"ID": IDlist, "TARGET": y_pred})
submission.to_csv("../Result/mxltendStackingXGB.csv", index=False)