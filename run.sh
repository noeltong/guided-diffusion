TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 64 --lr 2.5e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 64 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

mpiexec -n 4 --allow-run-as-root python scripts/classifier_train.py --data_dir /root/data/mice/npy $TRAIN_FLAGS $CLASSIFIER_FLAGS