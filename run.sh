#!/usr/bin/env bash

dataset=$1
seed=${2:-1}

python run-on-your-own-data.py --sample_size 10000 \
                               --dataset_fn data/$dataset.json \
                               --dataset $dataset \
                               --run_number $seed \
                               --gpu
