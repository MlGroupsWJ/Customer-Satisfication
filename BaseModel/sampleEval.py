# -*- coding:UTF-8 -*-
from Common.ModelCommon import *
from Common.EDACommon import *
from sklearn import linear_model, svm, tree, naive_bayes
from EDA.EDA3 import train_data, test_data, IDlist
import os


def unSampleEval(train, test, ratelist):
    for rate in ratelist:
        svc = svm.LinearSVC()
        train_data = underSampling(train, rate)
        X = train_data.iloc[:, :-1]
        y = train_data.iloc[:, -1]
        filename = '../Result/EDA3_SVC_UnSample_%d.csv' % ratelist.index(rate)
        SaveResult(svc, X, y, test, IDlist, filename)
        os.popen('kaggle competitions submit -c santander-customer-satisfaction -f %s -m "unsample_rate:%f"' % (filename, rate))


ratelist = np.arange(0.5, 5, 0.2).tolist()
# unSampleEval(train_data, test_data, ratelist)


def overSampleEval(train, test, repeatlist):
    for repeat in repeatlist:
        svc = svm.LinearSVC()
        train_data = overSampling(train, repeat)
        X = train_data.iloc[:, :-1]
        y = train_data.iloc[:, -1]
        filename = '../Result/EDA3_SVC_OverSample_%d.csv' % repeatlist.index(repeat)
        SaveResult(svc, X, y, test, IDlist, filename)
        os.popen('kaggle competitions submit -c santander-customer-satisfaction -f %s -m "oversample_repeat:%d"' % (filename, repeat))


repeatlist = list(range(15, 31))
# overSampleEval(train_data, test_data, repeatlist)