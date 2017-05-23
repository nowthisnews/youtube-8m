#!/usr/bin/env python3
import logging
import numpy as np
import os
import pickle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
class BaseModel:
    def init_datasets(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds
        
    def make(self, file_name):
        logging.info('Spliting dataset into training and test sets.')
        X_train, Y_train = self.prepare_dataset(self.train_ds)
        
        logging.info('Starting training %s...' % self.__class__.__name__)
        pretrained_model = self.find_parameters(X_train, Y_train)
        self.model = pretrained_model
        self.save_model(file_name)
        
    def prepare_dataset(self, dataset):
        features = np.array([object['features'] for object in dataset])
        labels = np.array([max(object['labels']) for object in dataset])
        return features, labels
    
    def find_parameters(self, X_train, Y_train, k_folds):
        pass
    
    def save_model(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.model.best_params_, file)
    
    def load_model(self, file_name):
        logging.info('Loading %s model from %s.', (self.__class__.__name__, file_name))
        self.model =  pickle.load(open(file_name, 'rb'))
        
    def evaluate_model(self):
        test_features, test_labels = self.prepare_dataset(self.test_ds)
        predicted = self.model.predict(test_features)
        
        assert test_labels.shape[0] == predicted.shape[0]
        
        result = 1 - np.count_nonzero(predicted-test_labels)/predicted.shape[0]
        self.result = result
        
    def save_results(self, directory):
        file_path = os.path.join(directory, "%s_results.txt" % self.__class__.__name__)
        with open(file_path, "w") as file:
            file.write("result: %s" % self.result)