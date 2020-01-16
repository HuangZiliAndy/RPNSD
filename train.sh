#!/bin/bash

stage=0

. path.sh

train_dir=data/swbd_sre_final/train_train
dev_dir=data/swbd_sre_final/train_dev
cfg=res101
cfg_file=cfgs/${cfg}.yml

# data process parameters
padded_len=20

# training parameters
epochs=1
batch_size=8
num_workers=4
optimizer=sgd
lr=0.01
min_lr=0.0001
scheduler=multi
patience=10
seed=7
alpha=1.0

# network parameters
arch=res101
large_scale=0
nclass=5963

# validate parameters
eval_interval=2000
num_dev=12000

exp_dir=experiment/cfg${cfg}epoch${epochs}bs${batch_size}op${optimizer}lr${lr}min_lr${min_lr}scheduler${scheduler}pat${patience}seed${seed}alpha${alpha}arch${arch}ls${large_scale}dev${num_dev}

if [ $stage -le 0 ]; then
    mkdir -p $exp_dir/{model,log} || exit 1;

    CUDA_VISIBLE_DEVICES=`free-gpu -n 1` scripts/train.py $exp_dir $train_dir \
	    $dev_dir --cfg_file $cfg_file --padded_len $padded_len \
	    --epochs $epochs --batch_size $batch_size --num_workers $num_workers --optimizer $optimizer \
	    --lr $lr --min_lr $min_lr --scheduler $scheduler --alpha $alpha \
	    --patience $patience --seed $seed --arch $arch --large_scale $large_scale \
	    --nclass $nclass --eval_interval $eval_interval --num_dev $num_dev \
	    --use_tfb > $exp_dir/log/train_log 
fi
