# -*- coding:UTF-8 -*-
from Data.data import *
from Common.EDACommon import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

# 针对float型变量画相关系数图
tmap = getTypeMap(train_data)
train_float = train_data.select_dtypes(include=['float64'])
train_int = train_data.select_dtypes(include=['int64'])
floatColList = tmap['float64']
# 互信息
mf = mutual_info_classif(train_float.values, train_data.TARGET.values, n_neighbors=3, random_state=17 )
# print(mf)

highCorrList = getHighCorrList(train_data[floatColList], 0.9, False)
corrDropList = getDropHighCorrList(highCorrList)
train_data.drop(corrDropList, axis=1, inplace=True)
print("after drop high corr cols:", train_data.shape)
corr_heatmap(train_data, floatColList)