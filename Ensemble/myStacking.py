from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
from bestScore import *
from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn import ensemble


# KFold顺序切分
def myStacking2(baseModelList, kfold, train_data, test_data, SecondModel):
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    kf = KFold(n_splits=kfold, shuffle=False, random_state=37)
    layer2_train = pd.DataFrame(np.zeros([train_data.shape[0], kfold]))
    layer2_test = pd.DataFrame(np.zeros([test_data.shape[0], kfold]))
    for i, baseModel in enumerate(baseModelList):
        train_predict = []
        test_predict = []
        for train_index, test_index in kf.split(x_train, y_train):
            subtrain_x = x_train.iloc[train_index, :]
            subtrain_y = y_train[train_index]
            subtest = x_train.iloc[test_index, :]
            baseModel.fit(subtrain_x, subtrain_y)
            subtrain_predict = baseModel.predict_proba(subtest)[:, -1]
            subtest_predict = baseModel.predict_proba(test_data)[:, -1]
            train_predict += subtrain_predict.tolist()
            test_predict.append(subtest_predict)
        test_predict_avg = np.mean(test_predict, axis=0)
        layer2_train[i] = train_predict
        layer2_test[i] = test_predict_avg
    SecondModel.fit(layer2_train, y_train)
    return SecondModel.predict_proba(layer2_test)[:, -1]


# StratifiedKFold打乱切分
def myStacking(baseModelList, kfold, train_data, test_data, SecondModel):
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    # StratifiedKFold切分保证训练集，测试集中各类别样本的比例与原始数据集中相同
    skf = StratifiedKFold(n_splits=kfold, shuffle=False, random_state=37)
    # step.1:求各baseModel对train data的fold之后的预测合并，以及对test data的预测，并存储进一个list
    layer2_train = pd.DataFrame(np.zeros([train_data.shape[0], kfold]))
    layer2_test = pd.DataFrame(np.zeros([test_data.shape[0], kfold]))
    for baseModel in baseModelList:
        '''
        对每个basemodel，定义两个list，train_predict存放每一次fold对subtrain的预测结果，全部fold跑完后再合并
        注意,这里用了StratifiedKFold,打乱切分,所以需要将subtrain的预测结果用对应index重组成df结构,用于最后的合并
        test_predict存放每一次fold对subtest的预测结果，全部fold跑完后再求平均
        '''
        train_predict = []
        test_predict = []
        for train_index, test_index in skf.split(x_train, y_train):
            # subtrain为第一层kfold切分出的train data，subtest同理
            subtrain_x = x_train.iloc[train_index, :]
            subtrain_y = y_train[train_index]
            subtest = x_train.iloc[test_index, :]

            baseModel.fit(subtrain_x, subtrain_y)
            # subtrain_predict：每一次fold对subtrain的预测结果，train_predict：kfold次subtrain的预测结果汇总
            subtrain_predict = baseModel.predict_proba(subtest)[:, -1]
            subtrain_predict = pd.DataFrame({"TARGET": subtrain_predict}, index=test_index)
            train_predict.append(subtrain_predict)
            # subtest_predict：每一次fold对test_data的预测结果，train_predict：kfold次test_data的预测结果汇总
            subtest_predict = baseModel.predict_proba(test_data)[:, -1]
            test_predict.append(subtest_predict)

        # step.2:每一个baseModel的kfold跑完之后，将kfold次subtrain的预测结果拼接成一个完整的对train的预测，kfold次test_data的预测结果求平均
        train_predict_merge = pd.concat(train_predict)
        test_predict_avg = np.mean(test_predict, axis=0)
        layer2_train[i] = train_predict_merge.values
        layer2_train.index = train_predict_merge.index
        layer2_test[i] = test_predict_avg

    # step.3:用第二层模型对上面得到的数据进行训练预测
    SecondModel.fit(layer2_train, y_train[layer2_train.index])
    return SecondModel.predict_proba(layer2_test)[:, -1]


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
abc = ensemble.AdaBoostClassifier()
gbc = ensemble.GradientBoostingClassifier()
clfs = [clf1, clf2, clf3]

lrc = linear_model.LogisticRegression(C=0.5, max_iter=300)
y_pred = myStacking(clfs, 5, train_data, test_data, lrc)
submission = pd.DataFrame({"ID": IDlist, "TARGET": y_pred})
submission.to_csv("../Result/myStackingXGB.csv", index=False)
# 经验证，加入异质模型的准确率反而下降，不如不同参数的XGB做BaseModel