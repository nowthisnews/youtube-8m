#!/usr/bin/env python3
from base_model import BaseModel
import logging
import pickle
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NTMLP(BaseModel):
    def find_parameters(self, X_train, Y_train):
        params = {
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.01],
                'learning_rate': ['invscaling', 'adaptive']
                }
        logging.info('Searching for the best parameters...')
        gs = GridSearchCV(MLPClassifier(probability=True), params, n_jobs = 10)
        pretrained_model = gs.fit(X_train, Y_train)
        
        return pretrained_model
    
    
if __name__=='__main__':
    with open('../train/train.p', 'rb') as file:
        train_ds = pickle.load(file)
    with open('../test/test.p', 'rb') as file:
        test_ds = pickle.load(file)
    mlp = NTMLP(train_ds, test_ds)
    mlp.make('mlp_model.p')