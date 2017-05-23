#!/usr/bin/env python3
from base_model import BaseModel
import logging
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NTGradientBoostingClassifier(BaseModel):
    def find_parameters(self, X_train, Y_train):
        # search the best svr hyperparameters
        parameters = {
                      'loss': ['deviance', 'exponential'],
                      'learning_rate': [0.001, 0.1, 1],
                      'n_estimators': [50, 100, 150, 200],
        }
        
        logging.info('Searching for the best parameters...')
        clf = GridSearchCV(GradientBoostingClassifier(), parameters, n_jobs = 10)
        pretrained_model = clf.fit(X_train, Y_train)
        
        return pretrained_model
    
if __name__=='__main__':
    with open('../train/train.p', 'rb') as file:
        train_ds = pickle.load(file)
    with open('../test/test.p', 'rb') as file:
        test_ds = pickle.load(file)
    tree = NTGradientBoostingClassifier(train_ds, test_ds)
    tree.make('gradient_boosting_model.p')