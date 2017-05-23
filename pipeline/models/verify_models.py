#!/usr/bin/env python3
import numpy as np
import pickle
from svm import NTSVM
from adaboost_classifier import NTAdaBoostClassifier
from decission_tree import NTDecisionTreeClassifier
from gaussian_classifier import NTGaussianProcessClassifier
from gradient_boosting import NTGradientBoostingClassifier
from k_neighbors import NTKNeighbors
from logistic_regression import NTLogisticRegression
from mlp import NTMLP
from sgd_classifier import NTSGDClassifier



class Verifier:
    def __init__(self, train_file_name, test_file_name):
        self.train = train_file_name
        self.test = test_file_name
        
    def open_sets(self):
        with open(self.train, 'rb') as file:
            self.train_ds = pickle.load(file)
        with open(self.test, 'rb') as file:
            self.test_ds = pickle.load(file)
    
    def clear_cache(self):
        self.model = None
            
    def train_model(self, model, directory):
        model.init_datasets(self.train_ds, self.test_ds)
        file_name = "%s_model.p" % model.__class__.__name__
        model.make(file_name)
        self.model = model
    
    def evaluate_model(self):
        self.model.evaluate_model()
    
    def save_results(self, directory):
        self.model.save_results(directory)
        

if __name__ == '__main__':
    train_ds = ""
    test_ds = ""
    save_model_dir = ""
    save_results_directory = ""
    verifier = Verifier(train_ds, test_ds)
    verifier.open_sets()
    models = [NTSVM(), NTAdaBoostClassifier(), NTDecisionTreeClassifier(),
              NTGaussianProcessClassifier(), NTGradientBoostingClassifier(), 
              NTKNeighbors(), NTLogisticRegression(), NTMLP(),
              NTSGDClassifier()]
    for model in models:
        verifier.train_model(model, save_model_dir)
        verifier.evaluate_model()
        verifier.save_results(save_results_directory)
        verifier.clear_cache()