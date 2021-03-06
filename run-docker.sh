# set defaults
USE_GPU=true
HOST_PORT=9999
CPU_IMAGE=docker-registry.nowth.is/yt8m:cpu
GPU_IMAGE=docker-registry.nowth.is/yt8m:gpu

DATA=/mnt/data/
MODELS=/mnt/models
NAME=video-tags

# parse arguments
while getopts gd:m:p:n:e: option
do 
    case "${option}" in 
        g) USE_GPU='true';;
        d) DATA=${OPTARG};; 
        m) MODELS=${OPTARG};; 
        p) HOST_POST=${OPTARG};; 
	n) NAME=${OPTARG};;
        *) error "unexpected option ${flag}";;
    esac
done

if $USE_GPU
then
    echo "Running docker on GPU"
    nvidia-docker run -it -d \
	-p $HOST_PORT:8888 \
	--log-driver=journald \
        --volume=$MODELS:/models \
        --volume=$DATA:/data \
	--name $NAME \
	--workdir=/workspace \
	--volume=$(pwd):/workspace $GPU_IMAGE /bin/bash
else
    echo "Running docker on CPU"
    docker run -it -d \
	    -p $HOST_PORT:8888 \
	    --log-driver=journald \
	    --volume=$MODELS:/models \
	    --volume=$DATA:/data \
	    --name $NAME \
	    --workdir=/workspace \
	    --volume=$(pwd):/workspace $CPU_IMAGE /bin/bash
fi
