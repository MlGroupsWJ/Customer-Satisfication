# -*- coding:UTF-8 -*-
from Data.data import *
from Common.EDACommon import *
from Common.ModelCommon import *
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


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
test_data.var3 = test_data.var3.replace(-999999, 2)

vardf = varShow(train_data)
# 方差小于0.001的直接丢弃
varabnormallist1 = vardf.feat[vardf['std'] < 0.001].tolist()
train_data.drop(varabnormallist1, axis=1, inplace=True)
print("after drop low var feat:", train_data.shape)

test_data.drop(zeroColumns, axis=1, inplace=True)
test_data.drop(repeatColumns, axis=1, inplace=True)
test_data.drop(varabnormallist1, axis=1, inplace=True)

features = selectFeatures(train_data, 70)
train_data = train_data[features + ['TARGET']]
test_data = test_data[features]

x_train = train_data.iloc[:, :-1]
y_train = train_data.TARGET


if __name__ == '__main__':
    # 原生XGB训练
    # params = {}
    # params['objective'] = 'binary:logistic'
    # params['booster'] = 'gbtree'
    # params['eval_metric'] = 'auc'
    # params['eta'] = 0.02
    # params['max_depth'] = 5
    # params['subsample'] = 0.6
    # params['colsample_bytree'] = 0.5
    #
    # d_train = xgb.DMatrix(x_train, label=y_train)
    # watchlist = [(d_train, 'train')]
    #
    # clf = xgb.train(params, d_train, 580, watchlist)
    #
    # d_test = xgb.DMatrix(test_data)
    # y_pred = clf.predict(d_test)
    # submission = pd.DataFrame({"ID": IDlist, "TARGET": y_pred})
    # submission.to_csv("../Result/bestXGB.csv", index=False)

    # xgb的sklearn接口XGBClassifier方法，需转换相关参数，经验证和原生xgb结果完全相同
    params = {}
    params['objective'] = 'binary:logistic'
    params['booster'] = 'gbtree'
    params['learning_rate'] = 0.02
    params['max_depth'] = 5
    params['subsample'] = 0.6
    params['colsample_bytree'] = 0.5
    params['n_estimators'] = 580

    clf = XGBClassifier(**params)
    clf.fit(x_train, y_train, eval_metric='auc')
    y_pred = clf.predict_proba(test_data)[:, -1]
    submission = pd.DataFrame({"ID": IDlist, "TARGET": y_pred})
    submission.to_csv("../Result/bestXGB.csv", index=False)
