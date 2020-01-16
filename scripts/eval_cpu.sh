#!/bin/bash

# This script computes region proposals, confidence score and speaker embeddings with RPNSD model

nj=40
stage=0
cmd="run.pl"
nclass=5963
cfg_file=cfgs/res101.yml
batch_size=1
num_workers=4
seed=7
arch=res101

. path.sh
. cmd.sh
. parse_options.sh || exit 1;

test_dir=$1
exp_dir=$2
modelname=$3
datasetname=$4

modelfile=$exp_dir/model/$modelname.pth.tar
output_dir=$exp_dir/result/$modelname/$datasetname

mkdir -p $output_dir/log || exit 1;
sdata=$test_dir/split$nj;
utils/split_data.sh $test_dir $nj || exit 1;

for i in $(seq $nj); do
  utils/filter_scp.pl -f 2 $sdata/$i/wav.scp $test_dir/rttm > $sdata/$i/rttm || exit 1;
done

# forward the network to get the predictions
if [ $stage -le 0 ]; then
    $cmd JOB=1:$nj $output_dir/log/inference.JOB.log \
	    python3 scripts/evaluate.py $exp_dir $sdata/JOB $modelfile \
	    --cfg_file $cfg_file --output_dir $output_dir/JOB --batch_size $batch_size --num_workers $num_workers \
	    --seed $seed --arch $arch \
	    --nclass $nclass --use_gpu 0 || exit 1;
fi

# merge predictions of different jobs
if [ $stage -le 1 ]; then
    python3 scripts/merge_prediction.py $output_dir $nj || exit 1;
fi
