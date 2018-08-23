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

params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'gbtree'
params['eval_metric'] = 'auc'
params['eta'] = 0.0201
params['max_depth'] = 5
params['subsample'] = 0.6815
params['colsample_bytree'] = 0.7
test_data.drop(zeroColumns, axis=1, inplace=True)
test_data.drop(repeatColumns, axis=1, inplace=True)
test_data.drop(varabnormallist1, axis=1, inplace=True)


# 寻找最佳的特征筛选比例
# plist = range(10, 110, 10)
# best_p, best_score = getBestP_and_AucScore(auc_score1, plist, params, train_data, 300, 5)
# print("bset_p is :%d, best_score is:%f" % (best_p, best_score))
# bset_p is :70, best_score is:0.841352

features = selectFeatures(train_data, 70)
train_data = train_data[features + ['TARGET']]
# ss = StandardScaler()
# train_data_x = train.iloc[:, :-1]
# train_data_y = train.iloc[:, -1]
# train_data_x = pd.DataFrame(data=ss.fit_transform(train_data_x), index=train_data_x.index, columns=train_data_x.columns)
# train_data = pd.concat([train_data_x, train_data_y], axis=1)
test_data = test_data[features]
# test_data = pd.DataFrame(ss.fit_transform(test_data), index=test_data.index, columns=test_data.columns)

x_train = train_data.iloc[:, :-1]
y_train = train_data.TARGET

# 增加各行0的统计
# x_train['n0'] = (x_train == 0).sum(axis=1)
# train_data['n0'] = x_train['n0']
# test_data['n0'] = (test_data == 0).sum(axis=1)
# train_data['n0'] = test_data['n0']

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

d_train = xgb.DMatrix(x_train, label=y_train)
watchlist = [(d_train, 'train')]

clf = xgb.train(params, d_train, 600, watchlist)

d_test = xgb.DMatrix(test_data)
y_pred = clf.predict(d_test)
submission = pd.DataFrame({"ID": IDlist, "TARGET": y_pred})
submission.to_csv("../Result/bestXGB.csv", index=False)