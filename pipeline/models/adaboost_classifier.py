#!/usr/bin/env python3
from base_model import BaseModel
import logging
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NTAdaBoostClassifier(BaseModel):
    def find_parameters(self, X_train, Y_train):
        # search the best svr hyperparameters
        parameters = {
                      'base_estimator': SVC(C=0.1, gamma=1e-08, kernel='linear'),
                      'n_estimators': [50, 100, 150, 200]
        }
        
        logging.info('Searching for the best parameters...')
        clf = GridSearchCV(AdaBoostClassifier(), parameters, n_jobs = 10)
        pretrained_model = clf.fit(X_train, Y_train, probability=True)
        
        return pretrained_model
    
if __name__=='__main__':
    with open('../train/train.p', 'rb') as file:
        train_ds = pickle.load(file)
    with open('../test/test.p', 'rb') as file:
        test_ds = pickle.load(file)
    tree = NTAdaBoostClassifier(train_ds, test_ds)
    tree.make('adaboost_model.p')