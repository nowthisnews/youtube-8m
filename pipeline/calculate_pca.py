"""
Usage: calculate_pca.py PATH OUTPUT [options]

Arguments:
    PATH    path to pickled features
    OUTPUT  path to where one would like to store calculated PCA matrix

Options:
    -l, --limit=<int>               Limit on the number of parsed files
    -c, --n-components=<int>        Number of ncomponents calculated in PCA
                                    [default: 1024]

    -s, --logging-interval=<int>    Log progress after this amount of steps
                                    [default: 100]

    --save-progress             If set to true dump pca matrix after number
                                of steps specified in `logging-step`
    --keep-scale                If set to true, do not perform whitening.

"""
import datetime
import glob
import itertools
import logging
import numpy as np
import os
import pickle
import time

from typeopt import Arguments
from sklearn.decomposition import IncrementalPCA

def batch(X, nrows=1024):
    '''
    X iterable with numpy arrays with variable shape along
    to first axis

    This function creates a generator that aggregates numpy arrays into
    bigger ones until they are big enough to be yield to IPCA.
    '''
    logger = logging.getLogger(__name__)

    res = None
    for x in X:
        if res is None:
            res = x
            continue

        if res.shape[0] > nrows:
            yield res
            res = x
        else:
            try:
                res = np.concatenate((res, x), axis=0)
            except ValueError as e:
                logger.exception("Shapes: %s, %s" % (res.shape, x.shape))
    yield res


def features(path, logging_step = 1000):
    logger = logging.getLogger(__name__)
    files = glob.glob(os.path.join(path, "*.pickle"))

    for index, file_path in enumerate(files):
        if index % logging_step == 0:
            logger.debug("Processed %d files, curent file: %s" % (index, file_path))
        with open(file_path, 'rb') as f:
            features, _ = pickle.load(f)

        if features.size == 0:
            logger.error("File %s has features with zero size! Skiping this file." % file_path)
            continue

        yield features

def save_progress(model, output_path, final=False):
    logger = logging.getLogger(__name__)


    if final:
        checkpoint_name = "final_ipca.pickle"
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        checkpoint_name = "ipca_%s.pickle" % timestamp

    checkpoint_path = os.path.join(output_path, checkpoint_name)

    logger.info("Saving model to %s" % checkpoint_path)
    with open(checkpoint_path, 'wb') as handle:
        pickle.dump(ipca, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    arguments = Arguments(__doc__, version='example 0.1')

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(arguments)

    ipca = IncrementalPCA(
        n_components=arguments.n_components,
        whiten=(not arguments.keep_scale))

    work_start_time = time.time()
    for i, x in enumerate(itertools.islice(batch(features(arguments.path)), arguments.limit), 1):
        batch_start_time = time.time()

        ipca.partial_fit(x,  check_input=False)
        current_time = time.time()
        if i % arguments.logging_interval == 0:
            logger.info("[%d] %s in %g (averge: %g)" % (
                i, x.shape, current_time - batch_start_time, (current_time - work_start_time)/i ))

            if arguments.save_progress:
                save_progress(ipca, arguments.output)

    save_progress(ipca, arguments.output, final=True)
