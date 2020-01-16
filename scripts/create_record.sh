#!/bin/bash

# The dataset is too large after data augmentation, even
# the mapping from utterance to feature consumes a lot of
# memory. So this script records the necessary information
# for each small file.

cmd="run.pl"
nj=40
stage=0
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 <data_dir>"
  echo "E.g.: $0 data/train"
  exit 1;
fi

data=$1

sdata=$data/split$nj

if [ $stage -le 0 ]; then
  utils/split_data.sh $data $nj || exit 1;
  
  for i in $(seq $nj); do
   utils/filter_scp.pl $sdata/$i/wav.scp $data/label.scp > $sdata/$i/label.scp
   utils/filter_scp.pl $sdata/$i/wav.scp $data/reco2dur > $sdata/$i/reco2dur
  done
  echo "Finish spliting data into $nj parts."
fi

if [ $stage -le 1 ]; then
  mkdir -p $data/data || exit 1;
  $cmd JOB=1:$nj $data/log/create_record.JOB.log \
    python3 scripts/create_record.py $sdata/JOB $data/data  
fi
