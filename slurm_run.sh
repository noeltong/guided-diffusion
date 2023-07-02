#!/bin/bash

#SBATCH --job-name=gd-train
#SBATCH --partition=normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mail-type=all
#SBATCH --mail-user=tongshq@shanghaitech.edu.cn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=sist_gpu62

TRAIN_FLAGS="--iterations 150000 --anneal_lr True --batch_size 32 --lr 2.5e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 64 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

mpiexec -n 4 --allow-run-as-root python scripts/classifier_train.py --data_dir /storage/data/tongshq/dataset/mice/npy $TRAIN_FLAGS $CLASSIFIER_FLAGS