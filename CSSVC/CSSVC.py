import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from util import model
from util import prediction
from sklearn.svm import SVC


class CSSVC(BaseEstimator, ClassifierMixin):

#Set parameters values
	def __init__(self,c = 1.0, d=3, g=0, r=0.0,m=100,t =0.001, h=1):


		self.c = c
		self.d = d
		self.g = g
		self.r = r
		self.m = m
		self.t = t
		self.h = h
		self.models_ = None

		
	def fit(self,X, y):

		"""
			Fit the model with the training data
			Parameters
			----------
			X: {array-like, sparse matrix}, shape (n_samples, n_features)
				Training patterns array, where n_samples is the number of samples
				and n_features is the number of features
			y: array-like, shape (n_samples)
				Target vector relative to X

			p: Label of the pattern which is choose for 1vsALL
			Returns
			-------
			self: object
			"""
		#Cargamos la configuracion del modelo de sklearn
		X,y = check_X_y(X,y)
		arg = ''
		argG = ''
		if(self.g == 0):
				argG = 'scale'

		elif(self.g == 1):
				argG = 'auto'
		
		#comenzamos con el proceso de fit 1VSAll
		classes = unique_labels(y)
		models = list()
		i = 0
		for label in classes:
			self.classifier_ = SVC(kernel='linear', C=self.c, degree=self.d, gamma=argG, coef0=self.r, cache_size=self.m, tol=self.t, shrinking=self.h) #configurados los parametros de clasificador
			labelModel = model()
			labelModel.label = label
			w = self.ordinalWeights(label, y)
			binarized_y = self.labelsBinarized(label,y)
			labelModel.classifier = self.classifier_.fit(X,binarized_y,sample_weight = w)
			models.insert(i,labelModel)
			i = i + 1
			
		self.models_ = models
		return self
   

	def predict(self,X): #el entrenamiento de los clasificadores se pierde al llegar aqui

		"""
		Performs classification on samples in X
		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
		Returns
		-------
		predicted_y : array, shape (n_samples, n_samples)
		Class labels for samples in X.
		"""
		#predictions = util.predicition
		
		check_is_fitted(self,'models_')
		preds = []
		decs = []
		X = check_array(X)
		for i in range(0,len(self.models_)):	
			print("prediction label", self.models_[i].label)

			pred  = self.models_[i].classifier.predict(X) #se obtienen las predicciones
			decv = self.models_[i].classifier.decision_function(X) #se obtienes los los valores de funcion de decision
			print("resultados",pred)
			preds.append(pred)
			print("valores de decision",decv)
			decs.append(decv)
		
		
		return preds , decs
		#return prediction(predicted_y, dec_values)

	
	
	def ordinalWeights(self,p, targets): # funciona bien
		w = np.array([],dtype = 'f')
		wp = np.array([],dtype = 'f')
		wDef = np.array([],dtype = 'f') 
		##print('clase target: ', p)
		#creamos los subconjunto con los elementos no pertencientes a la clase P y un subconjunto de 1 con tantos elementos como elementos pertencientes a la clase P en targets
		for iter in range(0,len(targets)):
			if(p != targets[iter]):
				aux = targets[iter]
				aux = abs(p-aux)
				aux = aux + 1
				w = np.append(w,aux) #vector subconjunto con las elementos distintos al elemento P
			else:
				wp= np.append(wp,1.00) #subconjunto de 1 
		#print('Subconjunto', w)
		subLen = len(w) #numero de elementos pertenecientes al subconjunto no P
		#print('SubLen: ', subLen)
		subSum = w.sum() #sumatorio  elementos subconjunto no P
		#print('subSum: ', subSum)
		w = w * subLen #multimplicamos cada elemento por el tamano del subconjunto
		#print('Multiplicacion: ', w)
		w = w/subSum   #dividimos el subconjunto por el sumatior de no p
		#print('Division : ', w)
		#juntamos los subconjuntos w y wp
		sp = 0
		noP = 0
		for iter in range(0,len(targets)):
			
			if(p != targets[iter]):
				wDef = np.append(wDef,w[noP])
				noP = noP + 1
			else:
				wDef = np.append(wDef,wp[sp])
				sp = sp + 1
		#print(wDef)
		return wDef
	
	def chooseWeight(self, preds, decs):
		decsDf = pd.DataFrame(decs)
		finalPreds = decsDf.idxmax(axis=0)
		finalPreds = finalPreds + 1
		return  finalPreds
        
        
	def labelsBinarized(self,p ,targets):
		w = np.array([],dtype = 'i')    		
		for iter in range(0,len(targets)):
			if(p != targets[iter]):
				w = np.append(w,0) #vector subconjunto con las elementos distintos al elemento P
			else:
				w = np.append(w,1) #subconjunto de 1 
        
		return w
		
       
