#!/usr/bin/env python3
from base_model import BaseModel
import logging
import pickle
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NTSGDClassifier(BaseModel):
    def find_parameters(self, X_train, Y_train):
        parameters = {
                      'loss': ['hinge', 'log', 'perceptron'], 
                      'learning_rate':['optimal', 'invscaling'],
                      'eta0':[0.0, 0.1, 1.0, 5.0, 10.0]
        }
        
        logging.info('Searching for the best parameters...')
        clf = GridSearchCV(SGDClassifier(), parameters, n_jobs = 10)
        pretrained_model = clf.fit(X_train, Y_train)
        
        return pretrained_model
    
if __name__=='__main__':
    with open('../train/train.p', 'rb') as file:
        train_ds = pickle.load(file)
    with open('../test/test.p', 'rb') as file:
        test_ds = pickle.load(file)
    sgd_classifier = NTSGDClassifier(train_ds, test_ds)
    sgd_classifier.make('logistic_regression_model.p')