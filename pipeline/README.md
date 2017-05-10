# Video tagging for videos stored in OVH
## Important scripts 
* `extract_features` - connects to psql database with videos and tags and prepares list for training data. Then uses ffmpeg to extract 1 frame per 1 second of video and extract inception_v3 features from frames. 
* `calculate_pca.py` - fits PCA transformation on extracted features to decrease dimensionality and saves it to disk

## Running
Use `run-docker.sh` to spin up the container and then run code inside the container
