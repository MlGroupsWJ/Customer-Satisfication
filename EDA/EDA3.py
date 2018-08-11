# -*- coding:UTF-8 -*-
from Data.data import *
from Common.EDACommon import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from math import ceil



def featShow(featlist):
    for feat in featlist:
        # cols = (int(len(featlist)/2) + 1) if len(featlist)%2 else int(len(featlist)/2)
        # cols = (lambda x: (int(x / 2) + 1) if x % 2 else int(x / 2))(len(featlist))
        cols = ceil(len(featlist)/2)
        ax = plt.subplot(2, cols, featlist.index(feat)+1)
        ax.scatter(range(train_data.shape[0]), train_data[feat].values, s=20)
        plt.xlabel('index')
        plt.ylabel(feat)
    plt.show()


def varShow(train_data):
    x = train_data[train_data.columns[:-1]]
    vardf = pd.DataFrame({'feat': x.columns, 'std': x.std().values})
    vardf_sort = vardf.sort_values(by='std')
    # print(vardf_sort)
    sns.barplot(data=vardf, x='feat', y='std')
    plt.xticks(rotation='vertical')
    plt.show()
    return vardf


pd.set_option('display.max_rows', 400, 'display.max_columns', 400)
# 全0值丢弃
zeroColumns = GetZeroColumns(train_data)
train_data.drop(zeroColumns, axis=1, inplace=True)
zerodf = GetZeroDf(train_data)

# 离群点丢弃
zerodf_sort = zerodf.sort_values(by='Count')
featlist = zerodf_sort[:7].index.tolist()
# featShow(featlist)
outlier1 = train_data.loc[train_data['num_var42_0'] > 100].index.tolist()
outlier2 = train_data.loc[train_data['num_var30_0'] > 100].index.tolist()
outlier3 = train_data.loc[(train_data['var38'] > 20000000)].index.tolist()
outlier = set(outlier1 + outlier2 + outlier3)
train_data.drop(outlier, axis=0, inplace=True)

vardf = varShow(train_data)
# 方差小于0.01的直接丢弃
varabnormallist1 = vardf.feat[vardf['std'] < 0.01].tolist()
train_data.drop(varabnormallist1, axis=1, inplace=True)
print("after drop low var feat:", train_data.shape)
# 方差值大的反常的单独分析
varabnormallist2 = vardf.feat[vardf['std'] > 100000000].tolist()
# featShow(varabnormallist2)
# valueCountsShow(train_data, varabnormallist2)
varabnormallist3 = vardf.feat[(vardf['std'] < 100000000) & (vardf['std'] > 10000000)].tolist()
# valueCountsShow(train_data, varabnormallist3)
# featShow(varabnormallist3)
# 分析发现存在多处1e+10的值，猜测代表某种异常值，类似全F，故用稍微远离其他样本值的值替代
train_data[train_data >= 9999999999] = 100
# varShow(train_data)

# 连续特征离散化,根据上图来区分划分区间
import warnings
warnings.filterwarnings("ignore")
train_data['var38'][train_data['var38'] <= 150000] = 0
train_data['var38'][(train_data['var38'] <= 225000) & (train_data['var38'] > 150000)] = 1
train_data['var38'][(train_data['var38'] <= 500000) & (train_data['var38'] > 225000)] = 2
train_data['var38'][(train_data['var38'] <= 1000000) & (train_data['var38'] > 500000)] = 3
train_data['var38'][(train_data['var38'] <= 6000000) & (train_data['var38'] > 1000000)] = 4
train_data['var38'][train_data['var38'] > 6000000] = 5

# 特征归一化，由于test_data可能会使用transform方法，直接使用train_data的拟合参数，所以这里必须将target去除再拟合
mm = MinMaxScaler()
train_data_x = train_data.iloc[:, :-1]
train_data_y = train_data.iloc[:, -1]
train_data_x = pd.DataFrame(data=mm.fit_transform(train_data_x), index=train_data_x.index, columns=train_data_x.columns)
train_data = pd.concat([train_data_x, train_data_y], axis=1)
print("after MinMaxScaler:", train_data.shape)

# 对比决策树feat_importance和xgb feat_importance，丢弃各自重要性很低的值
# 这里奇怪的是两种模型得到的importance倒数没有任何交叉
treeImptdf = TreeImportanceShow(train_data)
xgbImptdf = xgbImportanceShow(train_data)
imptdroplist1 = treeImptdf['feat'][treeImptdf['importance'] == 0].values.tolist()
imptdroplist2 = xgbImptdf['feature'][xgbImptdf['fscore'] < 10].values.tolist()
imptdroplist = list(set(imptdroplist1 + imptdroplist2))
train_data.drop(imptdroplist, axis=1, inplace=True)
print("after drop imptdroplist:", train_data.shape)

# undersampling
train_data = underSampling(train_data, 1.5)
print("after underSampling:", train_data.shape)

# 测试集的处理
test_data['var38'][test_data['var38'] <= 150000] = 0
test_data['var38'][(test_data['var38'] <= 225000) & (test_data['var38'] > 150000)] = 1
test_data['var38'][(test_data['var38'] <= 500000) & (test_data['var38'] > 225000)] = 2
test_data['var38'][(test_data['var38'] <= 1000000) & (test_data['var38'] > 500000)] = 3
test_data['var38'][(test_data['var38'] <= 6000000) & (test_data['var38'] > 1000000)] = 4
test_data['var38'][test_data['var38'] > 6000000] = 5
test_data.drop(zeroColumns, axis=1, inplace=True)
test_data[test_data >= 9999999999] = 100
test_data.drop(varabnormallist1, axis=1, inplace=True)
test_data = pd.DataFrame(mm.fit_transform(test_data), index=test_data.index, columns=test_data.columns)
test_data.drop(imptdroplist, axis=1, inplace=True)
print('test_date shape:', test_data.shape)
