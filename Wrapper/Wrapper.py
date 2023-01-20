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

    def __init__(self, base_classifier, scaler, labels=None , params=None):
        self.base_classifer = base_classifier
        self.labels = labels
        self.params = params
        self.scaler = scaler
        self.scaler_ = None
        self.classifier_ = None
    def fitPredict(self, dataTrain, dataTest):

        """
		Wrap fit and predict, makes the data usable for the methods, print metrics and result

		Parameters
		----------

		dataTrain: {pandas DataFrame}, shape (n_samples, n_features)
			Training patterns array, where n_samples is the number of
			samples and n_features is the number of features.

		dataTest: {pandas DataFrame}, shape (n_samples, n_features)
			Training patterns array, where n_samples is the number of
			samples and n_features is the number of features.

		Returns
		-------

		self: object
        """
        X_train = dataTrain.iloc[:, :-1].values
        y_train = dataTrain.iloc[:, -1].values
        X_test = dataTest.iloc[:, :-1].values
        y_test = dataTest.iloc[:, -1].values
        
        #discretizer = self.normalizeDataIntoCustomRange(y_train)
        #y_train = discretizer.transform(y_train.reshape(-1,1))
        self.clf.fit(X_train, y_train.ravel())
      #  y_testDiscretized = discretizer.transform(y_test.reshape(-1,1))
        prediction = self.clf.predict(X_test)
        prediction = prediction.round(decimals=0)
        prediction = np.absolute(prediction)
        score = accuracy_score(y_test,prediction)
        print(score)
       # print(self.showPredicitionLabelized(prediction))

        return self
  
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
        self.scaler_ = self.scaleData(y)
        self.y_ = self.scaler_.transform(y) 
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
        predicted_y = self.scaler_.inverse_transform(predicted_y.reshape(-1,1))
        predicted_y = predicted_y.round(decimals=0)
        predicted_y = np.absolute(predicted_y)
        return predicted_y
# Private methods

    def  normalizeDataIntoCustomRange(self,y):

        """
		Train the sklearn preprocessing model to transform the values of independent 
        variable from continous data to known nominal classes

		Parameters
		----------

		y: array-like, shape (n_samples)
			Target vector relative to X

		Returns
		-------

		discretizer : sklearn model
			
		"""

        if (self.labels is None):
            self.labels = np.array(["malo","regular","bueno","excelente"])
            discretizer = sklearn.preprocessing.KBinsDiscretizer(self.labels.size, encode='ordinal', strategy='uniform', dtype= np.float64)
            discretizer.fit(y.reshape(-1,1))
        else: 
            discretizer = sklearn.preprocessing.KBinsDiscretizer(self.labels.size, encode='ordinal', strategy='uniform', dtype= np.float64)
            discretizer.fit(y.reshape(-1,1))
        return discretizer

    def showPredicitionLabelized(self,prediction):

        """
		Create an array with the label for each Predicted Y

		Parameters
		----------

		prediction: array-like, shape (n_samples)
			Target vector relative to X

		Returns
		-------

		predictionLabelized : array-like, shape (n_samples)
			
		"""
        index = 0
        predictionLabelized = []
        for i in prediction:
            predictionLabelized.append(self.labels[int(i)])
        return predictionLabelized

    def scaleData (self, y_train):

        if(self.scaler == "Normalize"):
             scaler = sklearn.preprocessing.MinMaxScaler()
             scaler.fit(y_train)
             return scaler
        
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(y_train)
        return scaler
        





    




    