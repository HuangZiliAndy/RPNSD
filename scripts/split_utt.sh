#!/bin/bash

# This script splits long utterances into short segments
# for training efficiency. There is a tradeoff between
# disk space and training time.

cmd="run.pl"
nj=40
stage=0
uttlen=10.0
debug=0
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <data-dir> <data-split-dir>"
  echo "e.g.: $0 data/train data/train_10s"
  exit 1;
fi

data=$1
dir=$2

sdata=$data/split$nj

if [ $stage -le 0 ]; then
  utils/split_data.sh $data $nj || exit 1;
  
  for i in $(seq $nj); do
   utils/filter_scp.pl -f 2 $sdata/$i/wav.scp $data/rttm > $sdata/$i/rttm
   cp $data/spk2idx $sdata/$i/.
  done
  echo "Finish split data."
fi

if [ $stage -le 1 ]; then
  mkdir -p $dir/{log,tmp,wav,label} || exit 1;
  $cmd JOB=1:$nj $dir/log/split_utt.JOB.log \
    python3 scripts/split_utt.py $sdata/JOB $dir --uttlen $uttlen --debug $debug 
fi

if [ $stage -le 2 ]; then
  python3 scripts/prepare_wav_scp.py $dir || exit 1;
  python3 scripts/prepare_label_scp.py $dir || exit 1;
  awk -F' ' '{print $1, $1}' $dir/wav.scp > $dir/utt2spk || exit 1;
  awk -F' ' '{print $1, $1}' $dir/wav.scp > $dir/spk2utt || exit 1;
  utils/fix_data_dir.sh $dir || exit 1;
fi
