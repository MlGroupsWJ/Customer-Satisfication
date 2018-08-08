from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def ModelReport(estimator, modelname, train_data, k_fold):
    x = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]
    print("%s mean score:" % modelname, cross_val_score(estimator, x, y, cv=k_fold).mean())
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
