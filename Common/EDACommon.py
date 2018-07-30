# -*- coding:UTF-8 -*-
import pandas as pd
from minepy import MINE

class NAClass(object):
    def __init__(self):
        pass

    # 获取存在NA值的特征列表
    def GetNAFeatures(self, df):
        return df.columns[df.isnull().sum()!=0].tolist()

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
    zerodf.rename(columns={0: 'Count'},inplace=True)
    return zerodf


def GetZeroColumns(df):
    zeros = df[df != 0].count()
    return zeros[zeros == 0].index


def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic()
