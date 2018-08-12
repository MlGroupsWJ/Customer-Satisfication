# -*- coding:UTF-8 -*-
from Data.data import *
from Common.EDACommon import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


tmap = getTypeMap(train_data)
floatColList = tmap['float64']
print(len(floatColList))
corr_heatmap(train_data, floatColList[:10])