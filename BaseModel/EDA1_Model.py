from Common.ModelCommon import *
from sklearn import linear_model, svm, tree, naive_bayes
from EDA.EDA1 import train_data, test_data
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

lrc = linear_model.LogisticRegression()
# ModelCV(lrc, 'LogisticRegression', train_data, 5)
svc = svm.LinearSVC()
# ModelCV(svc, 'SVM', train_data, 5)
dtc = tree.DecisionTreeClassifier()
# ModelCV(dtc, 'DecisionTree', train_data, 5)
xgbc = XGBClassifier()
# ModelCV(xgbc, 'XGBClassifier', train_data, 5)


X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
IDlist = test_data.ID.values
test_data.drop('ID', axis=1, inplace=True)

# SaveResult(lrc, X, y, test_data, IDlist, '../Result/EDA1_LRC.csv')
# SaveResult(svc, X, y, test_data, IDlist, '../Result/EDA1_SVC.csv')
# SaveResult(dtc, X, y, test_data, IDlist, '../Result/EDA1_DTC.csv')
# SaveResult(xgbc, X, y, test_data, IDlist, '../Result/EDA1_XGBC.csv')
#
#
# ss = StandardScaler()
# X = ss.fit_transform(X)
# test_data = ss.transform(test_data)
#
SaveResult(lrc, X, y, test_data, IDlist, '../Result/EDA1_STD_LRC.csv')
SaveResult(svc, X, y, test_data, IDlist, '../Result/EDA1_STD_SVC.csv')
SaveResult(dtc, X, y, test_data, IDlist, '../Result/EDA1_STD_DTC.csv')
SaveResult(xgbc, X, y, test_data, IDlist, '../Result/EDA1_STD_XGBC.csv')

plot_learning_curve(lrc, '学习曲线', X, y)
