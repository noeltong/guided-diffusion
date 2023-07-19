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
#SBATCH --nodelist=sist_gpu66

MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2 --class_cond False --ema_rate 0.999,0.9999,0.9999432189950708"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 2.5e-4 --batch_size 32 --schedule_sampler loss-second-moment"

mpiexec -n 4 --allow-run-as-root python scripts/image_train.py --data_dir /storage/data/tongshq/dataset/mice/npy $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS