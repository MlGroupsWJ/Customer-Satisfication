from Common.ModelCommon import *
from sklearn import linear_model, svm, tree, naive_bayes, ensemble
from Ensemble.bestScore import train_data, test_data, IDlist
from xgboost import XGBClassifier

lrc = linear_model.LogisticRegression()
# ModelCV(lrc, 'lrc', train_data, 5)
# classReport(lrc, train_data)
svc = svm.LinearSVC(C=0.001)
# ModelCV(svc, 'SVM', train_data, 5)
# classReport(svc, train_data)
dtc = tree.DecisionTreeClassifier()
# ModelCV(dtc, 'DecisionTree', train_data, 5)
xgbc = XGBClassifier()
# ModelCV(xgbc, 'XGBClassifier', train_data, 5)
# classReport(xgbc, train_data)


etc = ensemble.ExtraTreesClassifier()
rfc = ensemble.RandomForestClassifier()
abc = ensemble.AdaBoostClassifier()
bgc = ensemble.BaggingClassifier()
gbc = ensemble.GradientBoostingClassifier()
rc = linear_model.RidgeClassifier()

X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

SaveResult(etc, X, y, test_data, IDlist, '../Result/EDA5_ETC.csv')
SaveResult(rfc, X, y, test_data, IDlist, '../Result/EDA5_RFC.csv')
SaveResult(abc, X, y, test_data, IDlist, '../Result/EDA5_ABC.csv')
SaveResult(gbc, X, y, test_data, IDlist, '../Result/EDA5_GBC.csv')
# SaveResult(lrc, X, y, test_data, IDlist, '../Result/EDA5_LRC.csv')
# SaveResult(svc, X, y, test_data, IDlist, '../Result/EDA5_SVC.csv')
# SaveResult(dtc, X, y, test_data, IDlist, '../Result/EDA5_DTC.csv')
# SaveResult(xgbc, X, y, test_data, IDlist, '../Result/EDA5_XGBC.csv')


# plot_learning_curve(lrc, '学习曲线', X, y)
