import argparse
import glob
import logging
import os
import pickle

from sklearn.model_selection import train_test_split

from pickle_dataset import PickleDataset
from tf_dataset import TensorDataset

def data_iterator(files, logging_step=1000):
    '''
    Given list of files loads data from pickled objects

    Arguments:
        files   - list of paths to pickles files
    Returns:
        Tuple with numpy array of size (nframes, 2048) and list with tags
    '''
    logger = logging.getLogger(__name__)

    for index, file_path in enumerate(files):
        video_id = os.path.basename(file_path).split(".")[0]
        if index % logging_step == 0:
            logger.debug(
                "Processed %d files, curent file: %s" % (index, file_path))

        with open(file_path, 'rb') as handle:
            features, tags = pickle.load(handle)

        if features.size == 0:
            logger.error(
                "File %s has features with zero size! Skiping this file." %
                file_path)
            continue

        yield video_id, features, tags

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data',
                        help = 'Path to data',
                        required = True)
    parser.add_argument('-i',
                        '--ipca',
                        help = 'Path to ipca',
                        required = True)
    parser.add_argument('-o',
                        '--output',
                        help = 'Path to the output file',
                        required = True)
    parser.add_argument('-p',
                        '--is_pickle',
                        help = 'Is save format pickle? T/F',
                        default = 'T',
                        required = False)
    parser.add_argument('-s',
                        '--split',
                        help = 'split',
                        required = False,
                        default=0.2)
    parser.add_argument('-l',
                        '--limit',
                        help = 'Path to data',
                        required = False,
                        default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)
    is_pickle = args.is_pickle
    is_pickle = True if is_pickle is 'T' or is_pickle is 'True' else False
    
    # get the list of files
    files = glob.glob(os.path.join(args.data,'*.pickle'))
    train, test = train_test_split(files, test_size=float(args.split))
    logger.info("Train size: %d, test size: %d" % (len(train), len(test)))

    # load ipca
    logger.info("Loading PCA from %s" % args.ipca)
    ipca = pickle.load(open(args.ipca, 'rb'))
    logger.info("Transforming train dataset")
    if is_pickle:
        transformer = PickleDataset(data_iterator=data_iterator, ipca=ipca)
        save_file_extension = '.p'
    else:
        transformer = TensorDataset(data_iterator=data_iterator, ipca=ipca)
        save_file_extension = '.tfrecord'
        
    
    transformer.make(
            os.path.join(args.output, 
                         'train/train%s' % save_file_extension),
            train, args.limit)
    logger.info("Transforming test dataset")
    transformer.make(
            os.path.join(args.output, 
                         'test/test%s' % save_file_extension),
            test, args.limit)
