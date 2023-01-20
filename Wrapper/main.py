import numpy as np
import pandas as pd 
import click
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from Wrapper import OrdinalWrapper
import json


@click.command()
@click.option('--train_file', '-T', default=None, required=True,
              help=u'Name of the file with training data.')
@click.option('--test_file', '-t', default=None, required=False,
              help=u'Fichero con los datos de test.')
def test_SVR(train_file, test_file):
	dataTrain = pd.read_csv("train_toy.txt", sep=' ',dtype=np.float64)
	dataTest = pd.read_csv("test_toy.txt", sep=' ',dtype=np.float64)
	X_train = dataTrain.iloc[:, :-1].values
	y_train = dataTrain.iloc[:, -1].values
	X_test = dataTest.iloc[:, :-1].values
	y_test = dataTest.iloc[:, -1].values
	params = "SVRParameters.json"
	paramsJson = open(params)
	parametersDictionary = json.load(paramsJson)
	labels = np.array(["muy malo","malo","regular","bueno","excelente"])

	clf = OrdinalWrapper(base_classifier="sklearn.svm.SVR",scaler = "standarize",labels=labels,params=parametersDictionary)
	
	clf.fit(X_train,y_train)
	prediction = clf.predict(X_test)
	print(accuracy_score(y_test,prediction))


@click.command()
@click.option('--train_file', '-T', default=None, required=True,
              help=u'Name of the file with training data.')
@click.option('--test_file', '-t', default=None, required=False,
              help=u'Fichero con los datos de test.') 
def test_randomForest(train_file, test_file):
	dataTrain = pd.read_csv("train_toy.txt", sep=' ',dtype=np.float64)
	dataTest = pd.read_csv("test_toy.txt", sep=' ',dtype=np.float64)
	X_train = dataTrain.iloc[:, :-1].values
	y_train = dataTrain.iloc[:, -1].values
	X_test = dataTest.iloc[:, :-1].values
	y_test = dataTest.iloc[:, -1].values
	params = "RandomForestParameter.json"
	paramsJson = open(params)
	parametersDictionary = json.load(paramsJson)
	labels = np.array(["muy malo","malo","regular","bueno","excelente"])

	clf = OrdinalWrapper(base_classifier="sklearn.ensemble.RandomForestRegressor",labels=labels,params=parametersDictionary)
	clf.fit(X_train,y_train)
	prediction = clf.predict(X_test)
	print(accuracy_score(y_test,prediction))


@click.command()
@click.option('--train_file', '-T', default=None, required=True,
              help=u'Name of the file with training data.')
@click.option('--test_file', '-t', default=None, required=False,
              help=u'Fichero con los datos de test.') 
def test_ridgeRegression(train_file,test_file):
	dataTrain = pd.read_csv("train_toy.txt", sep=' ',dtype=np.float64)
	dataTest = pd.read_csv("test_toy.txt", sep=' ',dtype=np.float64)
	X_train = dataTrain.iloc[:, :-1].values
	y_train = dataTrain.iloc[:, -1].values
	X_test = dataTest.iloc[:, :-1].values
	y_test = dataTest.iloc[:, -1].values
	params = "RidgeParameter.json"
	paramsJson = open(params)
	parametersDictionary = json.load(paramsJson)
	labels = np.array(["muy malo","malo","regular","bueno","excelente"])

	clf = OrdinalWrapper(base_classifier="sklearn.linear_model.Ridge",labels=labels,params=parametersDictionary)
	clf.fit(X_train,y_train)
	prediction = clf.predict(X_test)
	print(accuracy_score(y_test,prediction))

#test_randomForest()
#test_ridgeRegression()
test_SVR()