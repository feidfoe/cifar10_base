#!/bin/bash

GPUID=0
ARCH=vgg
DEPTH=16
DATA_SPLIT=train


dataset_path=<path/to/dataset>
checkpoint_path=<path/to/checkpoint>

nvidia-docker run -v `pwd`:`pwd` \
                  -v ${dataset_path}:`pwd`/dataset/ \
                  -v ${checkpoint_path}:`pwd`/checkpoint/ \
                  -w `pwd` \
                  --rm \
                  -it -d \
                  --ipc=host \
                  -e CUDA_VISIBLE_DEVICES=${GPUID} \
                  --name 'cifar10_base_gpu'${GPUID} \
                  feidfoe/pytorch:v.2 \
                  python main.py -e cifar10_base_${ARCH}${DEPTH} --arch ${ARCH} --depth ${DEPTH} --data_split ${DATA_SPLIT} --is_train --cuda
