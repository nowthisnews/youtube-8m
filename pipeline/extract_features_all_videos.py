import glob
import logging
import os
import pickle
import time

import mp
import sq

from db import fetch, inner_join, filter_videos
from utils.logging import setup_logging

MODEL_PATH = '/models/image/inception/classify_image_graph_def.pb'
DATA_PATH = '/data/video/video-level-features/mp/'
NPROD = 1
LIMIT = None
MIN_TAGS = 10
LOGGING_INTERVAL = 20

VQUERY = "select post_id, url from videos where status='ok'"
TQUERY = "select id, tags from videos where tags is not NULL"
TAGS = "select tag_id, name, path from content_tags"


def run_and_measure(fun, n):
    logger = logging.getLogger(__name__)
    start_time = time.time()
    fun()
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info("Elapsed time was %g seconds [%g]" % (elapsed, elapsed/n))


def main():
    ''' main '''
    setup_logging()
    logger = logging.getLogger(__name__)

    host='192.95.32.117'

    vdbname='ds-wizards'
    vuser='wizard'
    vpassword='GaG23jVxZhMnQaU53r8o'

    tdbname='ds-content-tags'
    tuser='ds-content-tags'
    tpassword='0fXjWl592vNf1gYvIw8w'

    vres = fetch(host, vdbname, vuser, vpassword, VQUERY)
    vres = [(post_id.split("_")[1], url) for post_id, url in vres]

    tres = fetch(host, tdbname, tuser, tpassword, TQUERY)
    videos = inner_join(tres, vres)

    filtered, t2i, i2t = filter_videos(videos, MIN_TAGS)
    logger.info("Found %d videos with %d unique tags" % (len(filtered), len(t2i)))

    # we will need this eventually
    tags = {
        tag_id: (name, path) for (tag_id, name, path) in fetch(
            host, tdbname, tuser, tpassword, TAGS)
    }

    # dump tags to file
    with open(os.path.join(DATA_PATH, "tags.pickle"), 'wb') as handle:
        pickle.dump(tags, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # check if we have processed some of those items
    processed = set([os.path.basename(f).split(".")[0]
                            for f in glob.glob(os.path.join(DATA_PATH, "*.pickle"))])

    logger.info("Found %d processed videos" % len(processed))

    filtered = [(video_id, tags, url) for (video_id, tags, url) in filtered if
                video_id not in processed]

    logger.info("After removing processed videos, %d items left" % len(filtered))

    fsq = lambda : sq.fetch(filtered[:LIMIT],
                            model_path=MODEL_PATH,
                            data_path=DATA_PATH,
                            logging_step=LOGGING_INTERVAL)

    fmp = lambda : mp.fetch(filtered[:LIMIT], nprod=NPROD,
                            model_path=MODEL_PATH,
                            data_path=DATA_PATH,
                            logging_step=LOGGING_INTERVAL)

    # Use only the first GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run_and_measure(fsq, len(filtered[:LIMIT]))


if __name__ == '__main__':
    main()
