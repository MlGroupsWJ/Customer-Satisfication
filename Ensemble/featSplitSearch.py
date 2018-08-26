# -*- coding:UTF-8 -*-
from Data.data import *
from Common.EDACommon import *
from Common.ModelCommon import *
import seaborn as sns
import matplotlib.pyplot as plt


def varShow(train_data):
    x = train_data[train_data.columns[:-1]]
    vardf = pd.DataFrame({'feat': x.columns, 'std': x.std().values})
    vardf_sort = vardf.sort_values(by='std')
    # print(vardf_sort)
    sns.barplot(data=vardf, x='feat', y='std')
    plt.xticks(rotation='vertical')
    # plt.show()
    return vardf

pd.set_option('display.max_rows', 400, 'display.max_columns', 400)
# 全0列丢弃
zeroColumns = GetZeroColumns(train_data)
train_data.drop(zeroColumns, axis=1, inplace=True)
zerodf = GetZeroDf(train_data)


# 重复列丢弃
C = train_data.columns.tolist()
repeatColumns = []
for i, c1 in enumerate(C):
    f1 = train_data[c1].values
    for j, c2 in enumerate(C[i+1:]):
        f2 = train_data[c2].values
        if np.all(f1 == f2):
            repeatColumns.append(c2)

train_data.drop(repeatColumns, axis=1, inplace=True)

# var3的特殊值用众数替代
train_data.var3 = train_data.var3.replace(-999999, 2)

vardf = varShow(train_data)
# 方差小于0.001的直接丢弃
varabnormallist1 = vardf.feat[vardf['std'] < 0.001].tolist()
train_data.drop(varabnormallist1, axis=1, inplace=True)
print("after drop low var feat:", train_data.shape)

test_data.drop(zeroColumns, axis=1, inplace=True)
test_data.drop(repeatColumns, axis=1, inplace=True)
test_data.drop(varabnormallist1, axis=1, inplace=True)

params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'gbtree'
params['eval_metric'] = 'auc'
params['eta'] = 0.02
params['max_depth'] = 3
params['min_child_weight'] = 5
params['subsample'] = 0.7
params['colsample_bytree'] = 0.5
params['early_stopping_rounds'] = 30

# 寻找最佳的特征筛选比例
plist = range(60, 80, 1)
best_p, best_score = getBestP_and_AucScore(auc_score1, plist, params, train_data, 600, 5)
print("bset_p is :%d, best_score is:%f" % (best_p, best_score))
# 第一次查找(10, 110, 10)：
# selectFeature p is :10, auc score is:0.824673
# selectFeature p is :20, auc score is:0.824585
# selectFeature p is :30, auc score is:0.835903
# selectFeature p is :40, auc score is:0.836445
# selectFeature p is :50, auc score is:0.839674
# selectFeature p is :60, auc score is:0.839635
# selectFeature p is :70, auc score is:0.841259
# selectFeature p is :80, auc score is:0.840597
# selectFeature p is :90, auc score is:0.840645
# selectFeature p is :100, auc score is:0.840381
# bset_p is :70, best_score is:0.841259

# 第二次查找(60, 80, 1)：
# selectFeature p is :60, auc score is:0.839635
# selectFeature p is :61, auc score is:0.839263
# selectFeature p is :62, auc score is:0.839496
# selectFeature p is :63, auc score is:0.839638
# selectFeature p is :64, auc score is:0.839323
# selectFeature p is :65, auc score is:0.840245
# selectFeature p is :66, auc score is:0.840399
# selectFeature p is :67, auc score is:0.840312
# selectFeature p is :68, auc score is:0.840458
# selectFeature p is :69, auc score is:0.840700
# selectFeature p is :70, auc score is:0.841259
# selectFeature p is :71, auc score is:0.840669
# selectFeature p is :72, auc score is:0.840895
# selectFeature p is :73, auc score is:0.840759
# selectFeature p is :74, auc score is:0.840747
# selectFeature p is :75, auc score is:0.840947
# selectFeature p is :76, auc score is:0.840783
# selectFeature p is :77, auc score is:0.841001
# selectFeature p is :78, auc score is:0.840821
# selectFeature p is :79, auc score is:0.840706
# bset_p is :70, best_score is:0.841259