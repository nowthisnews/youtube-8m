#!/usr/bin/env python3
import logging
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
class BaseModel:
    def __init__(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds
        
    def make(self, k_folds, file_name):
        logging.info('Spliting dataset into training and test sets.')
        X_train, Y_train = self.prepare_dataset(self.train_ds)
        
        logging.info('Starting training %s...' % self.__class__.__name__)
        pretrained_model = self.find_parameters(X_train, Y_train, k_folds)
        self.model = pretrained_model
        self.save_model(file_name)
        
    def prepare_dataset(self, dataset):
        features = np.array([object['features'] for object in dataset])
        labels = np.array([object['label'] for object in dataset])
        return features, labels
    
    def find_parameters(self, X_train, Y_train, k_folds):
        pass
    
    def save_model(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.model.best_params_, file)
    
    def load_model(self, file_name):
        logging.info('Loading SVC model from %s.', file_name)
        self.model =  pickle.load(open(file_name, 'rb'))
        
    def evaluate_model(self):
        pass