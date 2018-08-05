from Data.data import *
from Common.EDACommon import *
import seaborn as sns
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt


def featShow(featlist):
    for feat in featlist:
        ax = plt.subplot(2, 4, featlist.index(feat)+1)
        ax.scatter(range(train_data.shape[0]), train_data[feat].values, s=20)
        plt.xlabel('index')
        plt.ylabel(feat)
    plt.show()


def varShow(train_data):
    x = train_data[train_data.columns[:-1]]
    vardf = pd.DataFrame({'feat': x.columns, 'std': x.std().values})
    vardf_sort = vardf.sort_values(by='std')
    print(vardf_sort)
    sns.barplot(data=vardf, x='feat', y='std')
    plt.xticks(rotation='vertical')
    plt.show()
    return vardf


pd.set_option('display.max_rows', 400, 'display.max_columns', 400)
zeroColumns = GetZeroColumns(train_data)
train_data.drop(zeroColumns, axis=1, inplace=True)
zerodf = GetZeroDf(train_data)
zerodf_sort = zerodf.sort_values(by='Count')
featlist = zerodf_sort[:7].index.tolist()
print(GetTargetDf(train_data, 'TARGET'), "\n")
featShow(featlist)
outlier1 = train_data.loc[train_data['num_var42_0'] > 100].index.tolist()
outlier2 = train_data.loc[train_data['num_var30_0'] > 100].index.tolist()
outlier3 = train_data.loc[(train_data['var38'] > 20000000)].index.tolist()
outlier = set(outlier1 + outlier2 + outlier3)
train_data.drop(outlier, axis=0, inplace=True)
vardf = varShow(train_data)
vardroplist1 = vardf.feat[vardf['std'] < 1].tolist()
vardroplist2 = vardf.feat[vardf['std'] > 10000000].tolist()
vardroplist = vardroplist1 + vardroplist2
train_data.drop(vardroplist, axis=1, inplace=True)
vardf = varShow(train_data)

