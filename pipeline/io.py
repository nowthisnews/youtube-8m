import hashlib
import logging
import os
import pickle

import tensorflow as tf


def cheap_hash(txt, length=11):
    '''
    Hashes a sting
    '''
    hash = hashlib.sha1()
    hash.update(txt.encode('utf-8'))
    return bytes(hash.hexdigest()[:length], 'utf-8')


def data_iterator(files, logging_step=1000):
    '''
    Given list of files loads data from pickled objects

    Arguments:
        files   - list of paths to pickles files
    Returns:
        Tuple with numpy array of size (nframes, 2048) and list with tags
    '''
    for index, file_path in enumerate(files):
        video_id = os.path.basename(file_path).split(".")[0]
        if index % logging_step == 0:
            logger.debug("Processed %d files, curent file: %s" %
                (index, file_path))

        with open(file_path, 'rb') as f:
            features, tags = pickle.load(f)

        if features.size == 0:
            logger.error(
                "File %s has features with zero size! Skiping this file." %
                file_path)
            continue

        yield video_id, features, tags


def create_example(video_id, labels, features):
    '''
    Creates tf example
    '''
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'video_id': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[cheap_hash(video_id)])),
                'labels': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=labels)),
                'mean_rgb': tf.train.Feature(
                    float_list=tf.train.FloatList(value=features))
            }))
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
    tfrecord_file_queue = tf.train.string_input_producer(
        filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    tfrecord_features = tf.parse_single_example(
        tfrecord_serialized,
        features={
            'video_id': tf.FixedLenFeature([], tf.string),
            'labels': tf.VarLenFeature(tf.int64),
            'mean_rgb': tf.FixedLenFeature([], tf.float32),
        }, name='features')

    return tfrecord_serialized
