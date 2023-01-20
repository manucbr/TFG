import numpy as np
import pandas as pd 
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from CSSVC import CSSVC
from util import model
    

def test_CSSVC():  
	#test_file == None
	#train_file  == None		
	#if(test_file == None):
		#data = pd.read_csv(train_file, sep=' ')
		#X = data.iloc[: :-1].values
		#y = data.iloc[: -1].values
	#else:
	dataTrain = pd.read_csv("train_toy.txt", sep=' ',dtype=np.float64)
	dataTest = pd.read_csv("test_toy.txt", sep=' ',dtype=np.float64)
	X_train = dataTrain.iloc[:, :-1].values
	y_train = dataTrain.iloc[:, -1].values
	X_test = dataTest.iloc[:, :-1].values
	y_test = dataTest.iloc[:, -1].values
	classes = unique_labels(y_train)
	clf = CSSVC()
	clf.fit(X_train,y_train)
	preds, decs = clf.predict(X_test)
	defPredictions = clf.chooseWeight(preds,decs)
	print(defPredictions)
	print(accuracy_score(y_test,defPredictions))
test_CSSVC()