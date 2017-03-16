#!/bin/bash

MODE=1
if [ $# -gt 0 ]; then
    if [ "$1" == "train" ]; then
        MODE=1
    elif [ "$1" == "test" ]; then
        MODE=2
    elif [ "$1" == "extract" ]; then
        MODE=3
    else
        echo "Usage: run.sh train|test|extract [model_file]"
        exit 0
    fi
fi
ROOT=../../

network=inception_resnet_v2_rnn
train_batch_size=32
TRAIN_DIR=UCF101_temporal_inception_resnet_v2_echonet
DATASET_DIR=/state/partition1/shiyemin/data/ucf101/frames_tvl1
NUM_CLASSES=101
labels_offset=1
n_steps=16
modality=flow
read_stride=5

if [ $MODE -eq 1 ]; then
    GPU_ID=8,9,10,11,12,13,14,15
    NUM_CLONES=7
    train_steps=35000

    # Where the pre-trained InceptionV3 checkpoint is saved to.
    PRETRAINED_CHECKPOINT=/home/shiyemin/code/tensorflow-models/models/inception_resnet_v2_2016_08_30.ckpt

    # Where the dataset is saved to.
    DATASET_LIST=/home/shiyemin/data/ucf101/prepared_list/trainlist03_list.txt

    # Fine-tune only the new layers for 1000 steps.
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${ROOT}train_video_classifier.py \
      --mode=train \
      --num_clones=${NUM_CLONES} \
      --train_dir=${TRAIN_DIR}/train \
      --dataset_list=${DATASET_LIST} \
      --dataset_dir=${DATASET_DIR} \
      --NUM_CLASSES=${NUM_CLASSES} \
      --n_steps=${n_steps} \
      --modality=${modality} \
      --read_stride=${read_stride} \
      --labels_offset=${labels_offset} \
      --resize_image_size=256 \
      --train_image_size=224 \
      --model_name=${network} \
      --checkpoint_path=${PRETRAINED_CHECKPOINT} \
      --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
      --max_number_of_steps=${train_steps} \
      --batch_size=${train_batch_size} \
      --learning_rate_decay_type=piecewise \
      --learning_rate=0 \
      --learning_rate_steps=0.0225,0.01,0.001,0.0001,0.00001,0.000001 \
      --decay_iteration=3000,22000,27000,30000,33000 \
      --label_smoothing=0.1 \
      --save_interval_secs=600 \
      --save_summaries_secs=60 \
      --log_every_n_steps=10 \
      --optimizer=rmsprop \
      --weight_decay=0.00004
elif [ $MODE -eq 2 ]; then
    # Where the dataset is saved to.
    DATASET_LIST=/home/shiyemin/data/ucf101/prepared_list/testlist03_list.txt

    # Fine-tune only the new layers for 1000 steps.
    CUDA_VISIBLE_DEVICES=7 python ${ROOT}train_video_classifier.py \
      --mode=test \
      --train_dir=${TRAIN_DIR}/train \
      --eval_dir=${TRAIN_DIR}/eval \
      --dataset_list=${DATASET_LIST} \
      --dataset_dir=${DATASET_DIR} \
      --NUM_CLASSES=${NUM_CLASSES} \
      --n_steps=${n_steps} \
      --modality=${modality} \
      --read_stride=${read_stride} \
      --labels_offset=${labels_offset} \
      --resize_image_size=256 \
      --train_image_size=224 \
      --model_name=${network} \
      --batch_size=128 \
      --top_k=1
else
    trap 'echo you hit Ctrl-C/Ctrl-\, now exiting..; pkill -P $$; exit' SIGINT SIGQUIT
    EXTRACT_DIR=${TRAIN_DIR}/features
    rm -rf $EXTRACT_DIR
    mkdir $EXTRACT_DIR
    GPUS=(8 9 10 11)
    for((i=0;i<${#GPUS[@]};i++));do
        GPU_ID=${GPUS[$i]}
        echo "Using GPU "$GPU_ID
        # Where the dataset is saved to.
        DATASET_LIST=/home/shiyemin/data/ucf101/prepared_list/split3/testlist03_0$i
        # Fine-tune only the new layers for 1000 steps.
        CUDA_VISIBLE_DEVICES=$GPU_ID python ${ROOT}train_video_classifier.py \
          --mode=extract \
          --train_dir=${TRAIN_DIR}/train \
          --dataset_list=${DATASET_LIST} \
          --dataset_dir=${DATASET_DIR} \
          --NUM_CLASSES=${NUM_CLASSES} \
          --n_steps=${n_steps} \
          --modality=${modality} \
          --read_stride=${read_stride} \
          --labels_offset=${labels_offset} \
          --resize_image_size=256 \
          --train_image_size=224 \
          --model_name=${network} \
          --feature_dir=$EXTRACT_DIR \
          --batch_size=160 \
          --top_k=1 &
        sleep 20
    done
    wait
fi
