"""
Usage: prepare_datasets.py DATA IPCA OUTPUT [options]

Arguments:
    PATH    path to pickled features
    OUTPUT  path to where one would like to store calculated PCA matrix

Options:
    -l, --limit=<int>               Limit on the number of parsed files

    -s, --logging-interval=<int>    Log progress after this amount of steps
                                    [default: 100]

    --split=<float>                 Train-test split
                                    [default: 0.8]
"""
import glob
import hashlib
import logging
import os
import pickle
import itertools
import numpy as np

import tensorflow as tf

from typeopt import Arguments
from sklearn.model_selection import train_test_split

def cheap_hash(txt, length=11):
    '''
    Hashes a sting
    '''
    hash = hashlib.sha1()
    hash.update(txt.encode('utf-8'))
    return bytes(hash.hexdigest()[:length], 'utf-8')


def create_example(video_id, labels, features):
    '''
    Creates tf example
    '''
    example = tf.train.Example(features=tf.train.Features(feature={
        'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[cheap_hash(video_id)])),
        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
        'mean_rgb': tf.train.Feature(float_list=tf.train.FloatList(value=features))
        }
    ))
    return example


def write_to_tfrecord(video_id, labels, features, output_file):
    '''
    Serializes featues as tf.train.Example protobuff object and stores it
    TFRecord file.

    Args:
        video_id - video filename or id
        labels   - list of tags, can be an empty list
        features - 1-D vector with video features
    '''
    writer = tf.python_io.TFRecordWriter(output_file)
    example = create_example(video_id, labels, features)
    writer.write(example.SerializeToString())
    writer.close()


def read_from_tfrecord(filenames):
     tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
     reader = tf.TFRecordReader()
     _, tfrecord_serialized = reader.read(tfrecord_file_queue)

     tfrecord_features = tf.parse_single_example(
         tfrecord_serialized,
         features={
             'video_id': tf.FixedLenFeature([], tf.string),
             'labels': tf.VarLenFeature(tf.int64),
             'mean_rgb': tf.FixedLenFeature([], tf.float32),
         },name='features')
     
     return tfrecord_serialized

def data_iterator(files, logging_step = 1000):

    for index, file_path in enumerate(files):
        if index % logging_step == 0:
            logger.debug("Processed %d files, curent file: %s" % (index, file_path))

        with open(file_path, 'rb') as f:
            features, tags = pickle.load(f)

        if features.size == 0:
            logger.error("File %s has features with zero size! Skiping this file." % file_path)
            continue

        yield features, tags

def transform_and_write(output_file, files, limit = 10):
    '''
    Transforms features and tags and write transformed examples
    into TFRecords

    Arguments:
    output_file     - dump path
    files           - data iterator returning np.array with
                      features (x, 2048) and list of tags
    limit           - limit the number of files
    '''
    writer = tf.python_io.TFRecordWriter(output_file)
    for i, (x, y) in enumerate(itertools.islice(data_iterator(files), limit)):
        x_transformed = np.mean(ipca.transform(x), axis=0)
        example = create_example(str(i), labels, x_transformed)
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    args = Arguments(__doc__, version='example 0.1')

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)

    # get the list of files
    files = glob.glob(os.path.join(args.data, "*.pickle"))
    train, test = train_test_split(files, test_size = args.split)
    logger.info("Train size: %d, test size: %d" % (len(train), len(test)))

    # load ipca
    logger.info("Loading PCA from %s" % args.ipca)
    ipca = pickle.load(open(args.ipca, 'rb'))

    logger.info("Transforming train dataset")
    transform_and_write(os.path.join(
        args.output_file, "train/train.tfrecord"), train, limit)

    logger.info("Transforming test dataset")
    transform_and_write(os.path.join(
        args.output_file, "test/test.tfrecord"), test, limit)
