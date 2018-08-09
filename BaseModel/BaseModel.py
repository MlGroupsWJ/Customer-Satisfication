from Common.ModelCommon import *
from sklearn import linear_model, svm, tree, naive_bayes
from EDA.EDA3 import train_data, test_data, IDlist
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

lrc = linear_model.LogisticRegression()
# ModelReport(lrc, 'LogisticRegression', train_data, 5)
svc = svm.LinearSVC()
# ModelReport(svc, 'SVM', train_data, 5)
dtc = tree.DecisionTreeClassifier()
# ModelReport(dtc, 'DecisionTree', train_data, 5)
xgbc = XGBClassifier()
# ModelCV(xgbc, 'XGBClassifier', train_data, 5)


X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

SaveResult(lrc, X, y, test_data, IDlist, '../Result/EDA3_LRC.csv')
SaveResult(svc, X, y, test_data, IDlist, '../Result/EDA1_SVC.csv')
SaveResult(dtc, X, y, test_data, IDlist, '../Result/EDA1_DTC.csv')
SaveResult(xgbc, X, y, test_data, IDlist, '../Result/EDA1_XGBC.csv')


# plot_learning_curve(lrc, '学习曲线', X, y)
