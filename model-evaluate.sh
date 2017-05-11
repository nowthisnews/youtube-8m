#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python eval.py  \
	--eval_data_pattern='/data/video/video-level-features/test/test*.tfrecord' \
	--model=LogisticModel\
	--train_dir=/models/video/logistic-nt/ \
	--run_once=True \
