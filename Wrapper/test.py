import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import binned_statistic
from sklearn.svm import SVR
import sklearn.preprocessing
import json
y = np.array([1.456, 7.2346, 5.56, 4.99, 6.023, 3.54, 10.11])
y = y.reshape(-1,1)
labelsName = np.array(["malo","regular","bueno","excelente"])
est = preprocessing.KBinsDiscretizer(labelsName.size, encode='ordinal', strategy='uniform', dtype= np.float64)
rangedY = est.fit_transform(y)
print(rangedY)





#print(pd.cut(y,labelsName.size,include_lowest=True))
#paramsJson = open("SVRParameters.json")
#parametersDictionary = json.load(paramsJson)
#print(parametersDictionary)
#clf = SVR()
#clf.set_params(**parametersDictionary)
#print(clf.get_params)
