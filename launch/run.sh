# ############################# TRAIN CLS ##############################

# TRAIN_FLAGS="--iterations 150000 --anneal_lr True --batch_size 64 --lr 2.5e-4 --save_interval 10000 --weight_decay 0.05"
# CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 64 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True"

# mpiexec -n 4 --allow-run-as-root python scripts/classifier_train.py --data_dir /storage/data/tongshq/dataset/mice/npy $TRAIN_FLAGS $CLASSIFIER_FLAGS


############################# TRAIN MODEL ##############################

MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 2 --class_cond False --ema_rate 0.999"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 2.5e-4 --batch_size 32 --schedule_sampler loss-second-moment"

mpiexec -n 4 python scripts/image_train.py --data_dir /storage/data/tongshq/dataset/mice/npy $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


# ############################# SAMPLE MICE ##############################

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --use_fp16 True --use_scale_shift_norm True"
# python classifier_sample.py $MODEL_FLAGS --classifier_scale 0.5 --classifier_path models/128x128_classifier.pt --model_path models/128x128_diffusion.pt $SAMPLE_FLAGS