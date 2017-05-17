#!/usr/bin/env python3
from base_model import BaseModel
import logging
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NTDecisionTreeClassifier(BaseModel):
    def find_parameters(self, X_train, Y_train):
        # search the best svr hyperparameters
        parameters = {
                      'max_features': ['auto', 'sqrt', 'log2']
        }
        
        logging.info('Searching for the best parameters...')
        clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs = 10)
        pretrained_model = clf.fit(X_train, Y_train)
        
        return pretrained_model
    
if __name__=='__main__':
    with open('../train/train.p', 'rb') as file:
        train_ds = pickle.load(file)
    with open('../test/test.p', 'rb') as file:
        test_ds = pickle.load(file)
    tree = NTDecisionTreeClassifier(train_ds, test_ds)
    tree.make('tree_model.p')