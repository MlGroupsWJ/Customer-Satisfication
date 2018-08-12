from Common.ModelCommon import *
from sklearn import linear_model, svm, tree, naive_bayes
from EDA.EDA3 import train_data, test_data, IDlist
from xgboost import XGBClassifier

lrc = linear_model.LogisticRegression()
# ModelCV(lrc, 'lrc', train_data, 5)
# classReport(lrc, train_data)
svc = svm.LinearSVC()
# ModelCV(svc, 'SVM', train_data, 5)
# classReport(svc, train_data)
dtc = tree.DecisionTreeClassifier()
# ModelCV(dtc, 'DecisionTree', train_data, 5)
xgbc = XGBClassifier()
# ModelCV(xgbc, 'XGBClassifier', train_data, 5)
classReport(xgbc, train_data)


X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

SaveResult(lrc, X, y, test_data, IDlist, '../Result/EDA3_LRC.csv')
SaveResult(svc, X, y, test_data, IDlist, '../Result/EDA3_SVC.csv')
SaveResult(dtc, X, y, test_data, IDlist, '../Result/EDA3_DTC.csv')
SaveResult(xgbc, X, y, test_data, IDlist, '../Result/EDA3_XGBC.csv')


# plot_learning_curve(lrc, '学习曲线', X, y)
