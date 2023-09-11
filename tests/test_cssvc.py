from sys import path as syspath
from os import path as ospath
import unittest

import numpy as np
import numpy.testing as npt

from CSSVC import CSSVC

class TestCSSVC(unittest.TestCase):

	train_file = np.loadtxt("train.0")
	test_file = np.loadtxt("test.0")

	def test_cssvc_fit_correct(self):
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		X_test = self.test_file[:,0:(-1)]

	def test_cssvc_fit_not_valid_data(self):
        
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		classifiers = [CSSVC(c = 1.0, d=-3, g=0, r=0.0,m=100,t =0.001, h=1),
                        CSSVC(c = -1.0, d=3, g=0, r=0.0,m=100,t =0.001, h=1)]
		for classifier in classifiers:
				model = classifier.fit(X_train, y_train)
				self.assertIsNone(model, "The CSSVC fit method doesnt return Null on error")

	def test_cssvc_fit_not_valid_data(self):
        #Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]
		X_train_broken = self.train_file[0:(-1),0:(-2)]
		y_train_broken = self.train_file[0:(-1),(-1)]

		#Test execution and verification
		classifier = CSSVC(c = 1.0, d=3, g=0, r=0.0,m=100,t =0.001, h=1)
		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, y_train_broken)
				self.assertIsNone(model, "The CSSVC fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit([], y_train)
				self.assertIsNone(model, "The CSSVC fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, [])
				self.assertIsNone(model, "The CSSVC fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train_broken, y_train)
				self.assertIsNone(model, "The CSSVC fit method doesnt return Null on error")

	def test_cssvc_predict_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		classifier = CSSVC(c = 1.0, d=3, g=0, r=0.0,m=100,t =0.001, h=1)
		classifier.fit(X_train, y_train)

		#Test execution and verification
		with self.assertRaises(ValueError):
			classifier.predict([])

if __name__ == '__main__':
	unittest.main()


    
