#!/usr/bin/env python3
from base_dataset import BaseDataset
import hashlib
import tensorflow as tf

class TensorDataset(BaseDataset):
    
    def create_example(self, video_id, labels, features):
        def cheap_hash(txt, length=11):
            '''
            Hashes a sting
            '''
            hash_ = hashlib.sha1()
            hash_.update(txt.encode('utf-8'))
            return bytes(hash_.hexdigest()[:length], 'utf-8')
        '''
        Creates tf example in format compliant with yt8m starter code
        '''
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'video_id': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[cheap_hash(video_id)])),
                'labels': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=labels)),
                'mean_rgb': tf.train.Feature(
                    float_list=tf.train.FloatList(value=features))
                }))
        return example
    
    def write_example(self, writer, example):
        writer.write(example.SerializeToString())
    
    def save_results(self, writer):
        writer.close()
    
    def make_writer(self, save_file_name):
        return tf.python_io.TFRecordWriter(save_file_name)