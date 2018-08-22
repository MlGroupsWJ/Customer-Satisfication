# -*- coding:UTF-8 -*-
from Data.data import *
from Common.EDACommon import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Binarizer, scale
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, mutual_info_classif

import warnings
warnings.filterwarnings("ignore")

# NA检验，为False说明无缺失值
print(train_data.isnull().any().any())

pd.set_option('display.max_rows', 400, 'display.max_columns', 400)


# var3的特殊值用众数替代
train_data = train_data.replace(-999999, 2)

# 最后TARGET前增加一列，表示本行0的个数
X = train_data.iloc[:, :-1]
y = train_data.TARGET
X['n0'] = (X == 0).sum(axis=1)
train_data['n0'] = X['n0']

# 着重分析var38
train_data.var38.describe()
train_data.loc[train_data['TARGET'] == 1, 'var38'].describe()
train_data.var38.hist(bins=1000)
train_data.var38.map(np.log).hist(bins=1000)
train_data.var38.map(np.log).mode()
train_data.var38.value_counts()
train_data.var38[train_data['var38'] != 117310.979016494].mean()
train_data.loc[~np.isclose(train_data.var38, 117310.979016), 'var38'].value_counts()
train_data.loc[~np.isclose(train_data.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100)
# 根据上图，决定将var38作为一个二分类特征，取117310的为1，其余值为0
train_data['var38mc'] = np.isclose(train_data.var38, 117310.979016)  # 新增var38mc列，取值117310为True，其余值为False
train_data['logvar38'] = train_data.loc[~train_data['var38mc'], 'var38'].map(np.log)  # 对取值不为117310的var38列的值求对数
train_data.loc[train_data['var38mc'], 'logvar38'] = 0  # var38mc为True的，将对应的logvar38置为0
print('Number of nan in var38mc', train_data['var38mc'].isnull().sum())
print('Number of nan in logvar38', train_data['logvar38'].isnull().sum())

# var15是xgbimportance第一的，单独分析
sns.FacetGrid(train_data, hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend()
# plt.show()

# 探索var15和var38之间的关系
sns.FacetGrid(train_data, hue="TARGET", size=10) \
   .map(plt.scatter, "var38", "var15") \
   .add_legend()
# plt.show()
sns.FacetGrid(train_data, hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0, 120])
# plt.show()
# Exclude most common value for var38
sns.FacetGrid(train_data[~train_data.var38mc], hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0, 120])
# plt.show()
# What is distribution of the age when var38 has it's most common value ?
sns.FacetGrid(train_data[train_data.var38mc], hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend()
# plt.show()

# 根据卡方检验和anova中的f值筛选特征
p = 3
X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p)
selectChi2.fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p)
selectF_classif.fit(X, y)
# get_support获取的是一个bool列表，每个位置对应该特征是否被选中
chi2_selected = selectChi2.get_support()
chi2_selected_features = [f for i, f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(), chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [f for i, f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),f_classif_selected_features))
# 选取两种筛选方式选出的交叉特征
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [f for f, s in zip(X.columns, selected) if s]
print(features)
train_data = train_data[features+['TARGET']]

# 测试集处理
test_data = test_data.replace(-999999, 2)
test_data = test_data[features]

ss = StandardScaler()
train_data_x = train_data.iloc[:, :-1]
train_data_y = train_data.iloc[:, -1]
train_data_x = pd.DataFrame(data=ss.fit_transform(train_data_x), index=train_data_x.index, columns=train_data_x.columns)
train_data = pd.concat([train_data_x, train_data_y], axis=1)
test_data = pd.DataFrame(ss.fit_transform(test_data), index=test_data.index, columns=test_data.columns)
train_data = underSampling(train_data, 1.7)  
