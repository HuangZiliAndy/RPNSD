#!/bin/bash

. path.sh

cfg=res101
cfg_file=cfgs/${cfg}.yml

# training parameters
freeze=0
set_bn_fix=0
pretrain_expname=cfgres101epoch1bs8opsgdlr0.01min_lr0.0001schedulermultipat10seed7alpha1.0archres101dev12000
pretrain_modelname=modelbest
pretrain_model=experiment/${pretrain_expname}/model/${pretrain_modelname}.pth.tar 
epochs=10
batch_size=8
num_workers=4
optimizer=sgd
lr=0.00004
min_lr=0.00004
patience=10
seed=7
alpha=0.1

# network parameters
arch=res101
nclass=1284

# validate parameters
eval_interval=800

for fold_num in {1..5}; do
  exp_dir=experiment/pretrain${pretrain_expname}${pretrain_modelname}freeze${freeze}bnfix${set_bn_fix}cfg${cfg}epoch${epochs}bs${batch_size}op${optimizer}lr${lr}min_lr${min_lr}pat${patience}seed${seed}alpha${alpha}arch${arch}/$fold_num
  train_dir=data/callhome_10s_combined_5folds/$fold_num/train_dev_train
  dev_dir=data/callhome_10s_combined_5folds/$fold_num/train_dev_dev

  mkdir -p $exp_dir/{model,log} || exit 1;
  
  CUDA_VISIBLE_DEVICES=`free-gpu -n 1` scripts/train.py $exp_dir $train_dir $dev_dir --cfg_file $cfg_file \
	  --freeze $freeze --set_bn_fix $set_bn_fix --pretrain_model $pretrain_model \
  	  --epochs $epochs --batch_size $batch_size --num_workers $num_workers --optimizer $optimizer \
  	  --lr $lr --min_lr $min_lr --patience $patience --seed $seed \
  	  --arch $arch --alpha $alpha \
  	  --nclass $nclass --eval_interval $eval_interval \
  	  --use_tfb > $exp_dir/log/train_log 
done
