#!/usr/bin/env python3
from base_dataset import BaseDataset
import numpy as np
import pickle

class PickleDataset(BaseDataset):
    
    def create_example(self, video_id, labels, features):
        '''
        Creates tf example in a dictionary format
        '''
        example = {
                'video_id': video_id,
                'labels': labels,
                'features': features
                }
        return example
    
    def write_example(self, writer, example):
        writer.append(example)
    
    def save_results(self, writer):
        writer = np.array(writer)
        with open(self.save_file_name, 'wb') as file:
           pickle.dump(writer, file)
    
    def make_writer(self, save_file_name):
        self.save_file_name = save_file_name
        return []