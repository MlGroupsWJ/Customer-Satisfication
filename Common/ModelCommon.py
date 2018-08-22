from sklearn.model_selection import cross_val_score, learning_curve, train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import Binarizer, StandardScaler, scale
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
import xgboost as xgb


def ModelCV(estimator, modelname, train_data, k_fold):
    x = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]
    print("%s mean score:" % modelname, cross_val_score(estimator, x, y, cv=k_fold).mean())
    return cross_val_score(estimator, x, y, cv=k_fold).mean()

# TODO
def classReport(estimator, train_data):
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_data.TARGET, test_size=0.5, random_state=33)
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print(classification_report(y_test, y_predict))


def SaveResult(estimator, x, y, testdf, idlist, filename):
    estimator.fit(x, y)
    result = estimator.predict(testdf)
    resultdf = pd.DataFrame({'ID': idlist, 'TARGET': result.astype(np.int32)})
    resultdf.to_csv(filename, index=False)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(u"训练样本数")
    plt.ylabel(u"得分")
    plt.gca().invert_yaxis()
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

    plt.legend(loc="best")

    plt.draw()
    plt.savefig('../picture/learningcurve')
    plt.gca().invert_yaxis()
    plt.show()


# 通过xgb.cv做cv验证
def auc_score1(params, train_data, num_boost_round, kfold):
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    dtrain = xgb.DMatrix(x_train, y_train)
    cv_result = xgb.cv(params, dtrain, nfold=kfold, num_boost_round=num_boost_round, seed=123,
                       folds=StratifiedKFold(n_splits=kfold).split(x_train, y_train), metrics='auc')
    # 加不加StratifiedKFold有什么区别？
    # cv_result = xgb.cv(params, dtrain, nfold=kfold, num_boost_round=num_boost_round, metrics='auc')
    return cv_result.ix[num_boost_round-1, 'test-auc-mean']


# 通过StratifiedKFold切分数据，自己实现k-fold cv
def auc_score2(params, train_data, num_boost_round, kfold):
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    auc_score = 0
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1224)
    for train_index, eval_index in skf.split(x_train, y_train):
        train_set = x_train[train_index.astype(int).tolist()]
        eval_set = y_train[eval_index.astype(int).tolist()]
        train_set_x = train_set.iloc[:, :-1]
        train_set_y = train_set.iloc[:, -1]
        eval_set_x = eval_set.iloc[:, :-1]
        eval_set_y = eval_set.iloc[:, -1]

        dtrain = xgb.DMatrix(train_set_x, train_set_y)
        deval = xgb.DMatrix(eval_set_x)
        watchlist = [(dtrain, 'train')]
        xgb_model = xgb.train(params, dtrain, num_boost_round, watchlist)
        y_pred = xgb_model.predict_proba(deval)
        auc_score = roc_auc_score(y_pred, eval_set_y)
        return auc_score


def selectFeatures(train_data, p):
    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]
    X_bin = Binarizer().fit_transform(scale(X))
    selectChi2 = SelectPercentile(chi2, percentile=p)
    selectChi2.fit(X_bin, y)
    selectF_classif = SelectPercentile(f_classif, percentile=p)
    selectF_classif.fit(X, y)
    # get_support获取的是一个bool列表，每个位置对应该特征是否被选中
    chi2_selected = selectChi2.get_support()
    chi2_selected_features = [f for i, f in enumerate(X.columns) if chi2_selected[i]]
    # print('Chi2 selected {} features {}.'.format(chi2_selected.sum(), chi2_selected_features))
    f_classif_selected = selectF_classif.get_support()
    f_classif_selected_features = [f for i, f in enumerate(X.columns) if f_classif_selected[i]]
    # print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),f_classif_selected_features))
    # 选取两种筛选方式选出的交叉特征
    selected = chi2_selected & f_classif_selected
    # print('Chi2 & F_classif selected {} features'.format(selected.sum()))
    features = [f for f, s in zip(X.columns, selected) if s]
    return features


def getBestP_and_AucScore(auc_scoreFun, plist, params, train_data, num_boost_round, kfold):
    best_p = 0
    best_score = 0
    psdict = {}
    for p in plist:
        # 根据每一个p值进行特征选择，生成一份train data
        features = selectFeatures(train_data, p)
        train = train_data[features + ['TARGET']]
        ss = StandardScaler()
        train_data_x = train.iloc[:, :-1]
        train_data_y = train.iloc[:, -1]
        train_data_x = pd.DataFrame(data=ss.fit_transform(train_data_x), index=train_data_x.index,
                                    columns=train_data_x.columns)
        train = pd.concat([train_data_x, train_data_y], axis=1)

        # 对每一份train data进行cv，求其auc
        aucscore = auc_scoreFun(params, train, num_boost_round, kfold)
        psdict[p] = aucscore
        if aucscore > best_score:
            best_score = aucscore
            best_p = p
    for p, aucscore in psdict.items():
        print("selectFeature p is :%d, auc score is:%f" % (p, aucscore))
    return best_p, best_score
