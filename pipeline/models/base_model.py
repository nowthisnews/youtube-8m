#!/usr/bin/env python3
import logging
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, average_precision_score, \
                            precision_score, recall_score, \
                            cohen_kappa_score

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
        
        accuracy = accuracy_score(test_labels, predicted)
        average_precision = average_precision_score(test_labels, predicted)
        precision = precision_score(test_labels, predicted, average='micro')
        recall = recall_score(test_labels, predicted, average='micro')
        cohen_kappa = cohen_kappa_score(test_labels, predicted)
        
        self.result = {'accuracy': accuracy, 'average precision': average_precision,
                       'precision': precision, 'recall': recall,
                       'cohen kappa': cohen_kappa}
        
    def save_results(self, directory):
        file_path = os.path.join(directory, "%s_results.txt" % self.__class__.__name__)
        with open(file_path, "w") as file:
            for key, value in self.result.items():
                file.write("%s: %s" % (key, value))
            