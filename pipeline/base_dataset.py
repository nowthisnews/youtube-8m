#!/usr/bin/env python3
import numpy as np
import itertools

class BaseDataset:
    
    def __init__(self, data_iterator, ipca):
        self.data_iterator = data_iterator
        self.ipca = ipca
        
    def make(self, save_file_name, files, limit=None):
        writer = self.make_writer(save_file_name)
        for name, x, y in itertools.islice(self.data_iterator(files), limit):
            x_transformed = np.mean(self.ipca.transform(x), axis=0)
            example = self.create_example(name, y, x_transformed)
            self.write_example(writer, example)
            
        self.save_results(writer)
    
    def make_writer(self, save_file_name):
        pass
    
    def create_example(self, video_id, labels, features):
        pass
    
    def write_example(self, writer, example):
        pass
    
    def save_results(self, writer):
        pass