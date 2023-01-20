import numpy as np
import pandas as pd
from sklearn.svm import SVC
class util():
	def ordinalWeights(p, targets): #p --> clase que vamos a utilizar como positiva || targets --> vector con las clases de las tuplas del train
		count  = 0
		div = 0
		w = np.ones((targets.size,), dtype=int) #matriz de ceros para los pesos

		for i in range(len(targets)):
			if(targets[i] != p):
				div = abs(div + (p-targets[i]))
				count = count + 1

		for i in range(len(targets)):
			if(p != targets[i]):
				w[i] = (abs(p-targets[i])+1) * count / div+1
			else:
				w[i] = 1

		return w
		
class model():
	classifier = SVC()
	label = ''
	def __init__(self):
		self.classifier = None
		self.label = None
	
class prediction():
	pred = []
	decision_fuc = []

	def __init__(self, pred_y, decs):
		self.pred = pred_y
		self.decision_fuc = decs