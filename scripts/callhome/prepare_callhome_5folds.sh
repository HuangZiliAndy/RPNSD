#!/bin/bash

# This script prepare Callhome dataset with 5 fold cross validation
# Callhome dataset doesn't specify dev and test

. path.sh
. cmd.sh

stage=0

if [ $stage -le 0 ]; then
  # Prepare the Callhome portion of NIST SRE 2000.
  local/make_callhome.sh /export/corpora/NIST/LDC2001S97/ data/

  # There is some problem with the label file of iaeu.sph 
  mv data/callhome/fullref.rttm data/callhome/rttm || exit 1;
  sed -i '/iaeu/d' data/callhome/wav.scp || exit 1;
  sed -i '/iaeu/d' data/callhome/utt2spk || exit 1;
  sed -i '/iaeu/d' data/callhome/spk2utt || exit 1;
  utils/fix_data_dir.sh data/callhome || exit 1;

  python3 scripts/fix_rttm.py data/callhome/rttm data/callhome/rttm_new --add_uttname 1 || exit 1; 
  mv data/callhome/rttm data/callhome/rttm_bak || exit 1;
  mv data/callhome/rttm_new data/callhome/rttm || exit 1;
  python3 scripts/create_spk2idx.py data/callhome || exit 1;

  utils/data/get_utt2dur.sh data/callhome # 17.26 hours
  cp data/callhome/utt2dur data/callhome/reco2dur
fi
  
uttdur=10.0
if [ $stage -le 1 ]; then
  # split dataset into 10s chunks
  # prepare wav.scp, utt2dur, rttm and spk2idx before you split the dataset 
  scripts/split_utt.sh --cmd "$train_cmd" --nj 40 data/callhome data/callhome_10s || exit 1;
  awk -F' ' -v dur="$uttdur" '{print $1, dur}' data/callhome_10s/wav.scp > data/callhome_10s/reco2dur

  # data augmentation
  scripts/augmentation.sh --sample_rate 8000 --musan_dir /export/corpora/JHU/musan --prepare_musan false data/callhome_10s || exit 1;
  awk -F' ' -v dur="$uttdur" '{print $1, dur}' data/callhome_10s_combined/wav.scp > data/callhome_10s_combined/reco2dur
  # collect information for each 10s segment for training convenience
  scripts/create_record.sh --cmd "$train_cmd" --nj 10 data/callhome_10s_combined

  # split Callhome dataset into 5 folds
  python3 scripts/callhome/split_folds.py data/callhome_10s_combined data/callhome_10s_combined_5folds || exit 1;
  # split train/dev set again, so that we can compute speaker cross entropy loss on
  # the validation set
  python3 scripts/callhome/split_train_dev_callhome.py data/callhome_10s_combined_5folds 

  # we segment the utterances during training, but we want to use 
  # the whole length utterance during evaluation
  python3 scripts/callhome/prepare_whole_utt.py data/callhome_10s_combined_5folds data/callhome_5folds data/callhome || exit 1; 
  for i in {1..5}; do
    for dataset in train dev test; do
      awk -F' ' '{print $1, $1}' data/callhome_5folds/$i/$dataset/wav.scp > data/callhome_5folds/$i/$dataset/utt2spk || exit 1;
      awk -F' ' '{print $1, $1}' data/callhome_5folds/$i/$dataset/wav.scp > data/callhome_5folds/$i/$dataset/spk2utt || exit 1;
      utils/filter_scp.pl -f 2 data/callhome_5folds/$i/$dataset/wav.scp data/callhome/rttm > data/callhome_5folds/$i/$dataset/rttm || exit 1;
      utils/filter_scp.pl data/callhome_5folds/$i/$dataset/wav.scp data/callhome/reco2dur > data/callhome_5folds/$i/$dataset/reco2dur || exit 1;
      utils/filter_scp.pl data/callhome_5folds/$i/$dataset/wav.scp data/callhome/reco2num_spk > data/callhome_5folds/$i/$dataset/reco2num_spk || exit 1;
      utils/fix_data_dir.sh data/callhome_5folds/$i/$dataset || exit 1;
    done
  done

  for dataset in train dev test train_dev_train train_dev_dev; do
    for i in {1..5}; do
      ln -s `pwd`/data/callhome_10s_combined/data data/callhome_10s_combined_5folds/$i/$dataset/.
    done
  done
fi
