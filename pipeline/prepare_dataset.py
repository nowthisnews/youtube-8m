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
                                    [default: 0.2]
"""
import glob
import logging
import os
import pickle
import itertools
import numpy as np

import tensorflow as tf

from typeopt import Arguments
from sklearn.model_selection import train_test_split
from pipeline.io import create_example, data_iterator


def transform_and_write(output_file, files, limit=None):
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
    for name, x, y in itertools.islice(data_iterator(files), limit):
        x_transformed = np.mean(ipca.transform(x), axis=0)
        example = create_example(name, y, x_transformed)
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
    train, test = train_test_split(files, test_size=args.split)
    logger.info("Train size: %d, test size: %d" % (len(train), len(test)))

    # load ipca
    logger.info("Loading PCA from %s" % args.ipca)
    ipca = pickle.load(open(args.ipca, 'rb'))

    logger.info("Transforming train dataset")
    transform_and_write(os.path.join(
        args.output, "train/train.tfrecord"), train, args.limit)

    logger.info("Transforming test dataset")
    transform_and_write(os.path.join(
        args.output, "test/test.tfrecord"), test, args.limit)
