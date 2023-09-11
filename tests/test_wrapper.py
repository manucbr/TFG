from sys import path as syspath
from os import path as ospath
import unittest
import numpy as np
import numpy.testing as npt
from Wrapper import OrdinalWrapper
import json
class TestWRAPPER(unittest.TestCase):

	train_file = np.loadtxt("train.0")
	test_file = np.loadtxt("test.0")
    
	def test_wrapper_fit_correct(self):  #TODO
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		X_test = self.test_file[:,0:(-1)]

		expected_predictions = ["expectedPredictions.0", 
								"expectedPredictions.1",
								"expectedPredictions.2",]
		
		paramsSVR = "SVRParameters.json"
		paramsRandomForest = "RandomForestParameters.json"
		paramsRidge = "RidgeParameters.json"
		paramsJsonSVR = open(paramsSVR)
		paramsJsonRF = open(paramsRandomForest)
		paramsJsonRidge= open(paramsRidge)
		parametersSVRDictionary = json.load(paramsSVR)
		parametersRFDictionary = json.load(paramsJsonRF)    
		parametersRidgeDictionary = json.load(paramsJsonRidge)

		classifiers = [OrdinalWrapper(base_classifier="sklearn.svm.SVR",scaler = "Normalize",labels=None,params=parametersSVRDictionary),
						OrdinalWrapper(base_classifier="sklearn.ensemble.RandomForestRegressor",scaler = "Normalize",labels=None,params=parametersRFDictionary),
						OrdinalWrapper(base_classifier="sklearn.linear_model.Ridge",scaler = "Normalize",labels=None,params=parametersRidgeDictionary)]

		for expected_prediction, classifier in zip (expected_predictions, classifiers):
			classifier.fit(X_train, y_train)
			predictions = classifier.predict(X_test)
			expected_predictions = np.loadtxt(expected_prediction) 
			npt.assert_equal(predictions, expected_prediction, "The prediction doesnt match with the desired values")


	def test_wrapper_not_valid_parameter(self):
        
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]
		params = "SVRParameters.json"
		paramsErrorEpsilon = "SVRParametersErrorEpsilon.json"
		paramsErrorKernel = "SVRParametersErrorKernel.json"
		paramsErrorCoef = "SVRParemetersErrorCoef.json"
		paramsJson = open(params)
		paramsJsonEpsilon = open(paramsErrorEpsilon)
		paramsJsonKernel = open(paramsErrorKernel)
		paramsJsonCoef = open(paramsErrorCoef)
		parametersDictionary = json.load(paramsJson)
		parametersErrorEpsilonDictionary = json.load(paramsJsonEpsilon)    
		parametersErrorKernelDictionary = json.load(paramsJsonKernel)
		parametersErrorCoefDictionary = json.load(paramsJsonCoef)
		classifiers = [OrdinalWrapper(base_classifier="sklearn.svm.SVR",scaler = "Normalize",labels=None,params=parametersDictionary),
						OrdinalWrapper(base_classifier="sklearn.svm.SVR",scaler = "Normalize",labels=None,params=parametersErrorEpsilonDictionary),
						OrdinalWrapper(base_classifier="sklearn.svm.SVR",scaler = "Normalize",labels=None,params=parametersErrorKernelDictionary),
						OrdinalWrapper(base_classifier="sklearn.svm.SVR",scaler = "Normalize",labels=None,params=parametersErrorCoefDictionary)]
		
		error_msgs = ["parameters are correct",
						"Epsilon  value is invalid",
						"Kernel value is invalid",
						"Coef value is invalid"]

		for classifier, error_msgs in zip(classifiers, error_msgs):
				with self.assertRaisesRegex(ValueError, error_msgs):
					model = classifier.fit(X_train, y_train)
					self.assertIsNone(model, "The wrapper fit method doesnt return Null on error")

	def test_wrapper_fit_not_valid_data(self):
        #Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]
		X_train_broken = self.train_file[0:(-1),0:(-2)]
		y_train_broken = self.train_file[0:(-1),(-1)]
		params = "SVRParameters.json"
		paramsJson = open(params)
		parametersDictionary = json.load(paramsJson)
		#Test execution and verification
		classifier = OrdinalWrapper(base_classifier="sklearn.svm.SVR",scaler = "Normalize",labels=None,params=parametersDictionary)
		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, y_train_broken)
				self.assertIsNone(model, "The wrapper fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit([], y_train)
				self.assertIsNone(model, "The wrapper fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, [])
				self.assertIsNone(model, "The wrapper fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train_broken, y_train)
				self.assertIsNone(model, "The wrapper fit method doesnt return Null on error")

	def test_cssvc_predict_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]
		params = "SVRParameters.json"
		paramsJson = open(params)
		parametersDictionary = json.load(paramsJson)
		#Test execution and verification
		classifier = OrdinalWrapper(base_classifier="sklearn.svm.SVR",scaler = "Normalize",labels=None,params=parametersDictionary)
		classifier.fit(X_train, y_train)

		#Test execution and verification
		with self.assertRaises(ValueError):
			classifier.predict([])

if __name__ == '__main__':
	unittest.main()
