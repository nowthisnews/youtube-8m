#!/usr/bin/env python3
from base_model import BaseModel
import logging
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NTSVM(BaseModel):
    def find_parameters(self, X_train, Y_train):
        # search the best svr hyperparameters
        parameters = {
                      'kernel': ['rbf', 'linear'], 
                      'C':[1e-4, 1e-3, 0.1, 1, 10, 100], 
                      'gamma':[1e-8, 1e-7, 1e-6, 1e-4]
        }
        
        logging.info('Searching for the best parameters...')
        clf = GridSearchCV(SVC(), parameters, n_jobs = 10)
        pretrained_model = clf.fit(X_train, Y_train, probability=True)
        
        return pretrained_model
    
if __name__=='__main__':
    with open('../train/train.p', 'rb') as file:
        train_ds = pickle.load(file)
    with open('../test/test.p', 'rb') as file:
        test_ds = pickle.load(file)
    svm = NTSVM()
    svm.init_datasets(train_ds, test_ds)
    # svm.make('svc_model.p')
    my_svm = SVC()
    features, labels = svm.prepare_dataset(test_ds)
    my_svm.fit(features, labels)
    predicted = my_svm.predict(features)
    svm.model = my_svm
    svm.evaluate_model()
    svm.save_results('/Users/admin/youtube-8m/pipeline/models')