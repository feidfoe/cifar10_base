#!/bin/bash

GPUID=0
ARCH=resnet
DEPTH=50
DATA_SPLIT=test

if [ -z $1 ]; then
  step=349
else
  step=$1
fi
while [ ${#step} -lt 4 ];
do 
  step="0"$step 
done

CKPT=cifar10_base_${ARCH}${DEPTH}/checkpoint_step_${step}.pth


dataset_path=<path/to/dataset>
checkpoint_path=<path/to/checkpoint>



echo TEST MODEL : cifar10_base_${ARCH}${DEPTH}
echo CHECKPOINT : $CKPT
nvidia-docker run -v `pwd`:`pwd` \
                  -v ${dataset_path}:`pwd`/dataset/ \
                  -v ${checkpoint_path}:`pwd`/checkpoint/ \
                  -w `pwd` \
                  --rm \
                  -it \
                  --ipc=host \
                  -e CUDA_VISIBLE_DEVICES=${GPUID} \
                  --name 'cifar10_base_gpu'${GPUID} \
                  feidfoe/pytorch:v.2 \
                  python main.py -e cifar10_base_${ARCH}${DEPTH} --arch ${ARCH} --depth ${DEPTH} --data_split ${DATA_SPLIT} --checkpoint ${CKPT} --cuda

