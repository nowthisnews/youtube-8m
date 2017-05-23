#!/usr/bin/env python3
from base_model import BaseModel
import logging
import pickle
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NTKNeighbors(BaseModel):
     def find_parameters(self, X_train, Y_train):        
        params = {
                'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                'leaf_size': [30, 35, 40, 25],
                'p': [1, 2]
                }
        logging.info('Searching for the best parameters...')
        
        gs = GridSearchCV(KNeighborsClassifier(), params, n_jobs = 10)
        pretrained_model = gs.fit(X_train, Y_train, probability=True)
        
        return pretrained_model
    
if __name__=='__main__':
    with open('../train/train.p', 'rb') as file:
        train_ds = pickle.load(file)
    with open('../test/test.p', 'rb') as file:
        test_ds = pickle.load(file)
    svm = NTKNeighbors(train_ds, test_ds)
    svm.make('k_neighbors_model.p')