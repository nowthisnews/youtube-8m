#!/usr/bin/env python3
from adaboost_classifier import NTAdaBoostClassifier
import argparse
from decission_tree import NTDecisionTreeClassifier
from gaussian_classifier import NTGaussianProcessClassifier
from gradient_boosting import NTGradientBoostingClassifier
from k_neighbors import NTKNeighbors
import logging
from logistic_regression import NTLogisticRegression
from mlp import NTMLP
import os
import pickle
from sgd_classifier import NTSGDClassifier
from svm import NTSVM



class Verifier:
    def __init__(self, train_file_name, test_file_name):
        self.train = train_file_name
        self.test = test_file_name
        
    def open_sets(self):
        with open(self.train, 'rb') as file:
            self.train_ds = pickle.load(file)
        with open(self.test, 'rb') as file:
            self.test_ds = pickle.load(file)
    
    def clear_cache(self):
        self.model = None
            
    def train_model(self, model, directory):
        model.init_datasets(self.train_ds, self.test_ds)
        file_name = "%s_model.p" % model.__class__.__name__
        file_name = os.path.join(directory, file_name)
        model.make(file_name)
        self.model = model
    
    def evaluate_model(self):
        self.model.evaluate_model()
    
    def save_results(self, directory):
        self.model.save_results(directory)
        
        
def parse_arguments(parser):
    parser.add_argument('-t',
                        '--train_ds',
                        help = 'Path to train dataset',
                        required = True)
    parser.add_argument('-e',
                        '--test_ds',
                        help = 'Path to test dataset',
                        required = True)
    parser.add_argument('-m',
                        '--save_model_dir',
                        help = 'Model save directory',
                        required = True)
    parser.add_argument('-p',
                        '--save_results_directory',
                        help = 'Save results directory',
                        default = 'T',
                        required = True)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    train_ds = args.train_ds
    test_ds = args.test_ds
    save_model_dir = args.save_model_dir
    save_results_directory = args.save_results_directory
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)

    logger.info("Set models verifier")
    verifier = Verifier(train_ds, test_ds)
    logger.info("Open training and testing datasets")
    verifier.open_sets()
    logger.info("Create models")
    models = [NTSVM(), NTAdaBoostClassifier(), NTDecisionTreeClassifier(),
              NTGradientBoostingClassifier(), NTKNeighbors(), 
              NTLogisticRegression(), NTMLP(), NTSGDClassifier()]
    for model in models:
        logger.info("Train %s" % model.__class__.__name__)
        verifier.train_model(model, save_model_dir)
        logger.info("Evaluate %s" % model.__class__.__name__)
        verifier.evaluate_model()
        logger.info("Saving results")
        verifier.save_results(save_results_directory)
        logger.info("Clearing cache")
        verifier.clear_cache()
         
    logger.info("Done")