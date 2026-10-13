#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1 #,2,3,4,5,6,7

# settings
MODEL_ARC=$1
DATASET=$2
OUTPUT=results/${DATASET}-${MODEL_ARC}/
mkdir -p ${OUTPUT}

# CUDA_LAUNCH_BLOCKING=1
python3 -u train.py \
    --arch $MODEL_ARC \
    --train_list list/${DATASET}_list.txt \
    --workers 16 \
    --epochs 401 \
    --start-epoch 0 \
    --batch-size 200 \
    --learning-rate 1e-4 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --print-freq 100 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 1 \
    --parallel 1 \
    --pretrained results/imagenet-vgg16/imagenet-vgg16.pth \
    --dist-url 'tcp://localhost:10001' 2>&1 | tee ${OUTPUT}/output.log

    
    