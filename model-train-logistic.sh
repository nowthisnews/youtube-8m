#!/bin/bash
TRAIN_PATH=/models/video/logistic-nt
TRAIN_DATA_PATTERN='/data/video/video-level-features/train/train*.tfrecord'

CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_data_pattern=$TRAIN_DATA_PATTERN \
	--model=LogisticModel \
	--train_di$TRAIN_PATH
