#!/usr/bin/env python3
#!/usr/bin/env python3
from base_model import BaseModel
import logging
import pickle
from sklearn.gaussian_process import GaussianProcessClassifier



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NTGaussianProcessClassifier(BaseModel):
    def find_parameters(self, X_train, Y_train):
        # search the best svr hyperparameters
        logging.info('Fitting the model...')
        clf = GaussianProcessClassifier()
        pretrained_model = clf.fit(X_train, Y_train)
        
        return pretrained_model
    
if __name__=='__main__':
    with open('../train/train.p', 'rb') as file:
        train_ds = pickle.load(file)
    with open('../test/test.p', 'rb') as file:
        test_ds = pickle.load(file)
    svm = NTGaussianProcessClassifier(train_ds, test_ds)
    svm.make('svc_model.p')
