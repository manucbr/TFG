from symbol import parameters
from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
from utilities import load_classifier
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import binned_statistic

class OrdinalWrapper(BaseEstimator,ClassifierMixin):

    """
    Regression algorithms wrapper 

    The mainly purpose of this class is create a generic wrapper which could 
    obtains ordinal models by regression algorithms, the targets for the independent 
    variable could be provided by the users and it works all the regression algorithms 
    avaliable in sklearn.

   Parameters
   ------------

   classifier: sklearn regressor
       Base regressor used to build de model. this need to be a sklearn regressor.

    labels: String[]
       Array which include the labels choosed by the user to transform the continous 
       data into nominal data, if users does not specify the labels by himself the method
       will use a predefined values 
    
    params: String
       path of the Json file from where the method load the configuration for sklearn regressor
       in case of the user do not incluide it the regressor will use the defaoult value by sklearn.
       

    """

    def __init__(self, base_classifier, scaler = None,labels=None , params=None):
        self.base_classifer = base_classifier
        self.labels = labels
        self.params = params
        self.scaler = scaler 
        self.labels_ = None
        self.scaledY_ = None
        self.classifier_ = None

    def fit(self, X, y):

        """
		Fit the model with the training data and set the params for the regressor.

		Parameters
		----------

		X: {array-like, sparse matrix}, shape (n_samples, n_features)
			Training patterns array, where n_samples is the number of samples
			and n_features is the number of features

		y: array-like, shape (n_samples)
			Target vector relative to X

		Returns
		-------

		self: object
		"""
        X, y = check_X_y(X, y)
        self.X_ = X      
        if(None != self.params):
            estimator = load_classifier(self.base_classifer, self.params)
        y = y.reshape(-1,1)
        y  = self.maskYValues(y)
        self.scaledY_ = self.scaleData(y)
        self.y_ = self.scaledY_.transform(y.reshape(-1,1)) 
        estimator.fit(self.X_ ,self.y_.ravel())
        self.classifier_ = (estimator)
        return self

    def predict(self, X):

        """
		Performs classification on samples in X

		Parameters
		----------

		X : {array-like, sparse matrix}, shape (n_samples, n_features)

		Returns
		-------

		predicted_y : array, shape (n_samples,)
			Class labels for samples in X.
		"""

        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)
        predicted_y = self.classifier_.predict(X)
        predicted_y = self.scaledY_.inverse_transform(predicted_y.reshape(-1,1))
        predicted_y = self.roundToNearestClass(predicted_y)
        predicted_y = np.absolute(predicted_y)
        return predicted_y
# Private methods

        """
		Transform the y array with the labels given by the user

		Parameters
		----------

		Y : array-like, shape (n_samples)
			Target vector relative to X

		Returns
		-------

		NewLabels : array-like, shape (n_targets)
			
		"""

    def maskYValues(self,y):
        if(self.labels is None):
            self.labels_ = np.unique(y)
            return y
        self.labels_ = self.labels
        originalLabels = np.unique(y)
        newLabels = np.zeros(y.size)
        j = 0
        i = 0
        while i < y.size:
            while j < originalLabels.size:
                if(y[i] == originalLabels[j]):
                    newLabels[i] = self.labels[j]
                j = j + 1
            j = 0
            i = i + 1
        return newLabels

        """
        Train scaler object of sklearn library with the y data train and using the method
        of normalizing or stardandizing depend og the argument given by the user

		Parameters
		----------

		y_train : array-like, shape (n_samples)
			Target vector relative to X

		Returns
		-------

		scaler :  scaler, sklearn preprocessing object
			
		"""

    def scaleData (self, y_train):
        y_train = y_train.reshape(-1,1)
        if(self.scaler == "Normalize"):
             scaler = sklearn.preprocessing.MinMaxScaler()
             scaler.fit(y_train)
             return scaler
        
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(y_train)
        return scaler

        """
       Round the predictions to the closest value of y_test

		Parameters
		----------

		y_ : array-like, shape (n_samples)
			Target vector relative to X

		Returns
		-------

		roundY :  sarray-like, shape (n_samples)
			
		"""
    def roundToNearestClass (self, y):
        roundY = np.zeros(y.size)
        i = 0
        j = 1
        while i < y.size:
            roundY[i] = self.labels_[0]
            min = abs((self.labels_[0] - y[i]))
            while j < self.labels_.size:
                dif = abs( ( self.labels_[j] - y[i]))
                if( dif < min):
                    min = dif
                    roundY[i] = self.labels_[j]
                print(y[i])
                print(roundY[i])
                j = j + 1
            j = 1
            i = i + 1
        
        print(roundY)
        return roundY

    

        
    





    




    