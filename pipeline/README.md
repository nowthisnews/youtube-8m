# Video tagging for videos stored in OVH
## Important scripts 
* `extract_features_all_videos.py` - connects to psql database with videos and tags and prepares list for training data. Then uses ffmpeg to extract 1 frame per 1 second of video and extract inception_v3 features from frames. 
* `extract_featues_single_video.py`- given path to video extracts video-level features
* `calculate_pca.py` - fits PCA transformation on extracted features to decrease dimensionality and saves it to disk
* `prepare_dataset.py` - uses fitted PCA to decorelate incepction_v3 features extraced by `extract_features_all_videos.py` script, averages features per video and saves to `*.tfrecord` file

## Running
Use `run-docker.sh` to spin up the container and then run code inside the container
