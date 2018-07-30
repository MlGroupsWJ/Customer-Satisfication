# -*- coding:UTF-8 -*-
from Data.data import *
from Common.EDACommon import *
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold,SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier


# 特征值编号
featureList = train_data.columns.values
featureNo = list(range(train_data.shape[1]-1))
featureMap1 = dict(zip(featureNo, featureList))  # 特征->编号
featureMap2 = dict(zip(featureList, featureNo))  # 编号->特征
train_data.rename(columns=featureMap2, inplace=True)
target1_data_no = pd.DataFrame(train_data[train_data.TARGET == 1])

# 数据概况分析
pd.set_option('display.max_rows', 400, 'display.max_columns', 400)
# print(train_data.describe())
print("origrinal shape:", train_data.shape)
print(GetTargetDf(train_data, 'TARGET'), "\n")

# 全0值丢弃
zeroColumns = GetZeroColumns(train_data)
train_data.drop(zeroColumns, axis=1, inplace=True)
print(train_data.shape, "\n")

# 根据几个值较多的特征进行离群分析
zerodf = GetZeroDf(train_data)
zerodf_sort = zerodf.sort_values(by='Count')
pairplotlist = zerodf_sort[:7].index.tolist()
pairplotlist.append('TARGET')
print(pairplotlist)
# train_data[pairplotlist].to_csv('../csv/pairplot.csv')
# sns.pairplot(train_data[pairplotlist])
# plt.savefig('../picture/pairplot')
# plt.show()
# value_counts分析
# train_data[0].value_counts().plot.bar()
# plt.savefig('../picture/value_counts_0')
# train_data[1].value_counts().plot.bar()
# plt.savefig('../picture/value_counts')
# train_data[62].value_counts().plot.bar()
# plt.savefig('../picture/value_counts_62')
# train_data[137].value_counts().plot.bar()
# plt.savefig('../picture/value_counts_137')
# train_data[157].value_counts().plot.bar()
# plt.savefig('../picture/value_counts_157')
# train_data[193].value_counts().plot.bar()
# plt.savefig('../picture/value_counts_193')
# train_data[368].value_counts().plot.bar()
# plt.savefig('../picture/value_counts_368')
plt.show()
# print("value counts of 0:\n", train_data[0].value_counts())
outlier1 = train_data.loc[train_data[157] > 100].index.tolist()
outlier2 = train_data.loc[train_data[137] > 100].index.tolist()
outlier3 = train_data.loc[(train_data[368] > 20000000)].index.tolist()
outlier = set(outlier1 + outlier2 + outlier3)
print("first outlier list:", outlier)
train_data.drop(outlier, axis=0, inplace=True)
print("after first drop outlier shape:", train_data.shape)


# 过滤0占比超过99%且在正样本中也是全0的特征
zeroColumns_tag1 = GetZeroColumns(target1_data_no)
zeroColumns_99 = zerodf[zerodf.Percent > 0.99].index
zeroColumns2 = [col for col in zeroColumns_tag1 if col in zeroColumns_99]
train_data.drop(zeroColumns2, axis=1, inplace=True)
print("after drop zeros shape:", train_data.shape)

# pearson相关系数
def pearsonShow(df):
    pearsonDf = pd.DataFrame({'feat': df.columns[:-1]})
    corrs = []
    pvals = []
    for feat in df.columns[:-1]:
        corrs.append(pearsonr(df[feat], df[df.columns[-1]])[0])
        pvals.append(pearsonr(df[feat], df[df.columns[-1]])[1])
    pearsonDf['corr'] = corrs
    pearsonDf['pval'] = pvals
    plt.subplot(211)
    sns.barplot(data=pearsonDf, x='feat', y='corr')
    plt.subplot(212)
    sns.barplot(data=pearsonDf, x='feat', y='pval')
    plt.xticks(rotation='vertical')
    plt.show()
    # print(pearsonDf)

pearsonShow(train_data)

# 方差分析及过滤
x = train_data[train_data.columns[:-1]]
vardf = pd.DataFrame({'feat': x.columns, 'std': x.std().values})
sns.barplot(data=vardf, x='feat', y='std')
plt.show()
print("value counts of 196:\n", train_data[196].value_counts())
print("value counts of 208:\n", train_data[208].value_counts())
outlier1 = train_data[(train_data[196] != 0) & (train_data[196] != -1) & (train_data[196] < 1000000)].index.tolist()
outlier2 = train_data[((train_data[208] < 0) & (train_data[208] > -1)) | (train_data[208] == 1)].index.tolist()
outlier = set(outlier1 + outlier2)
train_data.drop(outlier, axis=0, inplace=True)
print("after second drop outlier shape:", train_data.shape)
x = train_data[train_data.columns[:-1]]
vardf = pd.DataFrame({'feat': x.columns, 'std': x.std().values})
fig = plt.figure()
ax1 = fig.add_subplot(311)
# print(vardf)
sns.barplot(data=vardf, x='feat', y='std')
plt.xticks(rotation='vertical')

# train_data=VarianceThreshold(threshold=3).fit_transform(train_data)

# 决策树特征选取
y = train_data['TARGET']
clf = ExtraTreesClassifier()
clf.fit(x, y)
imptdf = pd.DataFrame({'feat': x.columns, 'importance': clf.feature_importances_})
ax2 = fig.add_subplot(312)
sns.barplot(data=imptdf, x='feat', y='importance')
plt.xticks(rotation='vertical')


# 最大信息系数
# micdf = pd.DataFrame({'feat': x.columns, 'mic': list(range(len(x.columns)))})
# for i in range(len(micdf)):
#     micdf.iloc[i, 1] = mic(x[micdf.iloc[i, 0]], y)
#
# micdf.to_csv('../csv/micdf.csv', index=False)
micdf = pd.read_csv('../csv/micdf.csv')
ax3 = fig.add_subplot(313)
sns.barplot(data=micdf, x='feat', y='mic')
plt.xticks(rotation='vertical')
plt.show()


# 建立综合评分机制来筛选数据
vardf['var_score'] = vardf['std'].rank()
imptdf['impt_score'] = imptdf['importance'].rank()
micdf['mic_score'] = micdf['mic'].rank()

alldf = pd.concat([vardf['feat'], vardf['var_score'], imptdf['impt_score'], micdf['mic_score']], axis=1)
alldf['score'] = alldf['var_score'] + alldf['impt_score'] + alldf['mic_score']
alldf.drop(['var_score', 'impt_score', 'mic_score'], axis=1, inplace=True)
# for i in range(len(alldf)):
#     alldf.iloc[i, 0] = featureMap1[alldf.iloc[i, 0]]
sns.barplot(data=alldf, x='feat', y='score')
plt.xticks(rotation='vertical')
plt.show()

alldf.sort_values(by='score', ascending=False, inplace=True)
print(alldf)
drop_columns = alldf[-20:].feat.values
train_data.drop(drop_columns, axis=1, inplace=True)
print("final shape:", train_data.shape)
train_data.to_csv('../Data/train_eda1.csv')
