#!/usr/bin/env python3
from base_model import BaseModel
import logging
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NTLogisticRegression(BaseModel):
    def find_parameters(self, X_train, Y_train):
        parameters = {
                      'penalty': ['l2'], 
                      'C':[1e-4, 1e-3, 0.1, 1, 10, 100],
                      'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag']
        }
        
        logging.info('Searching for the best parameters...')
        clf = GridSearchCV(LogisticRegression(), parameters, n_jobs = 10)
        pretrained_model = clf.fit(X_train, Y_train)
        
        return pretrained_model
    
if __name__=='__main__':
    with open('../train/train.p', 'rb') as file:
        train_ds = pickle.load(file)
    with open('../test/test.p', 'rb') as file:
        test_ds = pickle.load(file)
    logistic_regression = NTLogisticRegression(train_ds, test_ds)
    logistic_regression.make('logistic_regression_model.p')