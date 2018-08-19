# -*- coding:UTF-8 -*-
import pandas as pd
from minepy import MINE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
import operator
from sklearn.utils import shuffle
from Common.ModelCommon import ModelCV
from sklearn import svm
import numpy as np


class NAClass(object):
    def __init__(self):
        pass

    # 获取存在NA值的特征列表
    def GetNAFeatures(self, df):
        return df.columns[df.isnull().sum() != 0].tolist()

    # 缺失特征按从多到少排序进行展示
    def ShowNAInfo(self, df, NAlist):
        NA_count = df[NAlist].isnull().sum().sort_values(ascending=False)
        NAInfo = pd.DataFrame({'NA_count': NA_count, 'NA_percent': NA_count/df.shape[0]})
        print(NAInfo)

    # 含缺失值特征处理的通用接口，strategy为处理策略
    def HandleNA(self, df, NAfeaturesList, strategy='mean'):
        if strategy == 'mean':
            for feature in NAfeaturesList:
                if df[feature].dtypes == 'object':
                    raise ValueError('Nonnumeric feature!')
                df[feature].fillna(df[feature].mean(), inplace=True)
        elif strategy == 'mode':
            for feature in NAfeaturesList:
                df[feature].fillna(df[feature].mode()[0], inplace=True)
        elif strategy == 'drop':
            df.drop(NAfeaturesList, axis=1, inplace=True)
        else:
            for feature in NAfeaturesList:
                if (df[feature].dtypes == 'object' and type(strategy) != str) or (
                        df[feature].dtypes != 'object' and type(strategy) == str):
                    raise ValueError('Mismatched type!')
                df[feature].fillna(strategy, inplace=True)

    def checkNA(self, df):
        return df.isnull().sum().max()


def CategoricalList(df):
    return [attr for attr in df.columns if df.dtypes[attr] == 'object']


def NumericalList(df):
    return [attr for attr in df.columns if df.dtypes[attr] != 'object']


def GetTargetDf(df, target):
    targetdf = pd.DataFrame(df[target].value_counts())
    targetdf['Percent'] = targetdf[target]/df.shape[0]
    return targetdf


def GetZeroDf(df):
    zerodf = pd.DataFrame(df[df == 0].count())
    zerodf['Percent'] = zerodf[0]/df.shape[0]
    zerodf.rename(columns={0: 'Count'}, inplace=True)
    return zerodf


def GetValueCountDf(df):
    valueCountList = []
    for feat in df.columns:
        valueCountList.append(df[feat].value_counts().shape[0])
    valueCountDf = pd.DataFrame({'feat': df.columns, 'valueCount': valueCountList})
    return valueCountDf


def GetZeroColumns(df):
    zeros = df[df != 0].count()
    return zeros[zeros == 0].index


def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic()


def featShow(train_data, feat):
    plt.scatter(range(train_data.shape[0]), train_data[feat].values, s=20)
    plt.xlabel('index')
    plt.ylabel(feat)
    plt.show()


def TypeShow(train_data):
    dtype_df = train_data.dtypes.reset_index()
    dtype_df.columns = ["Count", "Column Type"]
    print(dtype_df.groupby("Column Type").aggregate('count').reset_index())


# 通过决策树获取特征重要性
def TreeImportanceShow(train_data):
    x = train_data[train_data.columns[:-1]]
    y = train_data['TARGET']
    clf = ExtraTreesClassifier()
    clf.fit(x, y.astype('int'))
    imptdf = pd.DataFrame({'feat': x.columns, 'importance': clf.feature_importances_})
    imptdf_sort = imptdf.sort_values(by='importance', ascending=False)
    # print("decision tree importance:\n", imptdf_sort)
    sns.barplot(data=imptdf_sort, x='feat', y='importance')
    plt.xticks(rotation='vertical')
    # plt.show()
    return imptdf_sort


def xgbImportanceShow(train_data):
    x = train_data[train_data.columns[:-1]]
    y = train_data['TARGET']
    dtrain = xgb.DMatrix(x, y)
    xgb_params = {"objective": "binary:logistic", "eta": 0.01, "max_depth": 8, "seed": 42, "silent": 1}
    model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    impt = model.get_fscore()
    impt = sorted(impt.items(), key=operator.itemgetter(1))
    imptdf = pd.DataFrame(impt, columns=['feature', 'fscore'])
    imptdf_sort = imptdf.sort_values(by='fscore', ascending=False)
    # print("xgb importance:\n", imptdf_sort)
    imptdf_sort.to_csv('../tmp/xgb_importance.csv', index=False)
    xgb.plot_importance(model, max_num_features=400, height=0.8)
    # plt.show()
    return imptdf_sort


def valueCountsShow(train_data, featlist):
    for feat in featlist:
        print(train_data[feat].value_counts())


# rate为希望采样后的0样本的个数为rate*1样本
def underSampling(train, rate):
    idx_0 = train[train['TARGET'] == 0].index
    idx_1 = train[train['TARGET'] == 1].index
    len_1 = len(train.loc[idx_1])
    undersample_idx_0 = shuffle(idx_0, random_state=37, n_samples=int(len_1*rate))
    idx_list = list(undersample_idx_0) + list(idx_1)
    train = train.loc[idx_list].reset_index(drop=True)
    return train


# repeat为重复样本1的次数
def overSampling(train, repeat):
    idx_1 = train[train['TARGET'] == 1].index
    i = 0
    while i < repeat:
        train = pd.concat([train, train.iloc[idx_1, :]], axis=0).reset_index(drop=True)
        i += 1
    return train


# 通过train_data的cv分数来作为评判标准，但是每种不同比率的sample，最终的样本数有一定不同，是否影响指标的客观准确性？
def getBestUnSamplingRate(train, ratelist):
    bestscore = 0
    bestrate = 0
    for rate in ratelist:
        svc = svm.LinearSVC()
        train_data = underSampling(train, rate)
        score = ModelCV(svc, 'svm', train_data, 5)
        print("rate :%f, score:%f" % (rate, score))
        if score > bestscore:
            bestscore = score
            bestrate = rate
    print("best rate :%f, best score:%f" % (bestrate, bestscore))
    return bestrate


def corr_heatmap(train, v):
    correlations = train[v].corr()
    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show()



def typeShow(train_data):
    print(train_data.dtypes.value_counts())


def getTypeMap(train_data):
    typeMap = {}
    typeMap['int64'] = train_data.dtypes[train_data.dtypes == 'int64'].index
    typeMap['float64'] = train_data.dtypes[train_data.dtypes == 'float64'].index
    return typeMap


# iswhole为True时代表是完整的数据集，需要将TARGET去除再求相关性，为False时代表已经是筛选后的列，不包含TARGET
def getHighCorrList(df, thres, iswhole):
    if iswhole:
        x = df.iloc[:, :-1]
    else:
        x = df
    corr = x.corr()
    index = corr.index[np.where(corr > thres)[0]]
    columns = corr.columns[np.where(corr > thres)[1]]
    highCorrList = [[index[i], columns[i]] for i in range(len(index)) if index[i] != columns[i]]
    uniqList = [[0, 0]]
    for i in range(len(highCorrList)):
        uniqCount = 0
        for j in range(len(uniqList)):
            if highCorrList[i][0] == uniqList[j][1] and highCorrList[i][1] == uniqList[j][0]:
                uniqCount += 1
        if uniqCount == 0:
            uniqList.append(highCorrList[i])
    del uniqList[0]
    return uniqList


def getDropHighCorrList(highList):
    dropList = []
    for item in highList:
        if item[0] in dropList:
            break
        if item[1] in dropList:
            break
        else:
            dropList.append(item[1])
    return dropList


def getUinqueCorrDf(train, threshold):
    cor_mat = train.corr()
    important_corrs = (cor_mat[abs(cor_mat) > threshold][cor_mat != 1.0]).unstack().dropna().to_dict()
    unique_important_corrs = pd.DataFrame(
        list(set([(tuple(sorted(key)), important_corrs[key]) for key in important_corrs])),
        columns=['attribute pair', 'correlation'])
    unique_important_corrs = unique_important_corrs.ix[abs(unique_important_corrs['correlation']).argsort()[::-1]]
    return unique_important_corrs