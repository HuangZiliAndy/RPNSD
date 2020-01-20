#!/bin/bash

# This script prepares SRE SWBD diarization data

. path.sh
. cmd.sh

stage=0
debug=0

# The default data directories are for CLSP grid only.
# Please modify them to your own directories.

# Mixer 6 Speech
mx6_dir=/export/corpora/LDC/LDC2013S03

# SRE Speech
# 2010 NIST Speaker Recognition Evaluation Test Set
sre10_test_dir=/export/corpora5/SRE/SRE2010/eval/
# 2008 NIST Speaker Recognition Evaluation Test Set
sre08_test_dir=/export/corpora/LDC/LDC2011S08
# 2008 NIST Speaker Recognition Evaluation Training Set Part 1
sre08_train_dir=/export/corpora/LDC/LDC2011S05
# 2004 NIST Speaker Recognition Evaluation
sre04_dir=/export/corpora/LDC/LDC2006S44
# 2005 NIST Speaker Recognition Evaluation Training Data
sre05_train_dir=/export/corpora/LDC/LDC2011S01
# 2005 NIST Speaker Recognition Evaluation Test Data
sre05_test_dir=/export/corpora/LDC/LDC2011S04
# 2006 NIST Speaker Recognition Evaluation Training Set
sre06_train_dir=/export/corpora/LDC/LDC2011S09
# 2006 NIST Speaker Recognition Evaluation Test Set Part 1
sre06_test_part1_dir=/export/corpora/LDC/LDC2011S10
# 2006 NIST Speaker Recognition Evaluation Test Set Part 2
sre06_test_part2_dir=/export/corpora/LDC/LDC2012S01

# SWBD Speech
# Switchboard Cellular Part 1 Audio
swbd_cellular_part1_dir=/export/corpora/LDC/LDC2001S13
# Switchboard Cellular Part 2 Audio
swbd_cellular_part2_dir=/export/corpora5/LDC/LDC2004S07
# Switchboard-2 Phase I
swbd2_phase1_dir=/export/corpora/LDC/LDC98S75
# Switchboard-2 Phase II
swbd2_phase2_dir=/export/corpora5/LDC/LDC99S79
# Switchboard-2 Phase III Audio
swbd2_phase3_dir=/export/corpora5/LDC/LDC2002S06

if [ $stage -le 0 ]; then
  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $mx6_dir data/

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  local/make_sre10.pl $sre10_test_dir data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $sre08_test_dir $sre08_train_dir data/

  # This prepares the older NIST SREs from 2004-2006.
  wget -P data/local/ http://www.openslr.org/resources/15/speaker_list.tgz
  tar -C data/local/ -xvf data/local/speaker_list.tgz
  sre_ref=data/local/speaker_list
  local/make_sre.pl $sre04_dir \
     04 $sre_ref data/sre2004
  local/make_sre.pl $sre05_train_dir \
    05 $sre_ref data/sre2005_train
  local/make_sre.pl $sre05_test_dir \
    05 $sre_ref data/sre2005_test
  local/make_sre.pl $sre06_train_dir \
    06 $sre_ref data/sre2006_train
  local/make_sre.pl $sre06_test_part1_dir \
    06 $sre_ref data/sre2006_test_1
  local/make_sre.pl $sre06_test_part2_dir \
    06 $sre_ref data/sre2006_test_2
  rm data/local/speaker_list.*

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test_1 data/sre2006_test_2 \
    data/sre08 data/mx6 data/sre10
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl $swbd_cellular_part1_dir \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl $swbd_cellular_part2_dir \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl $swbd2_phase1_dir \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl $swbd2_phase2_dir \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl $swbd2_phase3_dir \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  utils/combine_data.sh data/swbd_sre data/swbd data/sre
  utils/fix_data_dir.sh data/swbd_sre || exit 1; # 91224 utterances
fi

if [ $stage -le 1 ]; then
  # In this step, we keep the utterances whose both channels are in data/swbd_sre
  python3 scripts/swbd_sre/filter_2channel_utt.py data/swbd_sre data/swbd_sre_filtered
  utils/data/get_utt2dur.sh --nj 40 --cmd "$train_cmd" data/swbd_sre_filtered || exit 1;
  utils/fix_data_dir.sh data/swbd_sre_filtered || exit 1; # 62928 utterances
fi

if [ $stage -le 2 ]; then
  # Compute vad decision with pretrained VAD on Fisher. For details see README_SAD.txt. 
  # I modified the parameter segment-padding to 0 (default 0.2).
  if [ ! -d "exp/segmentation_1a/tdnn_stats_asr_sad_1a" ]; then
    wget http://kaldi-asr.org/models/4/0004_tdnn_stats_asr_sad_1a.tar.gz
    tar xvzf 0004_tdnn_stats_asr_sad_1a.tar.gz
  fi

  steps/segmentation/detect_speech_activity.sh --nj 40 --cmd "$train_cmd" \
    --extra-left-context 79 --extra-right-context 21 \
    --extra-left-context-initial 0 --extra-right-context-final 0 --segment-padding 0 \
    --frames-per-chunk 150 --mfcc-config conf/mfcc_hires.conf \
    data/swbd_sre_filtered exp/segmentation_1a/tdnn_stats_asr_sad_1a \
    mfcc vad_work_dir data/swbd_sre_filtered
fi

if [ $stage -le 3 ]; then
  # Create RTTM file based on SAD results
  mkdir -p data/swbd_sre_diarization || exit 1;	
  python3 scripts/swbd_sre/create_rttm.py data/swbd_sre_filtered_seg/segments data/swbd_sre_diarization
  sort -k2,2 -k4,4n data/swbd_sre_diarization/rttm > data/swbd_sre_diarization/rttm_tmp
  mv data/swbd_sre_diarization/rttm_tmp data/swbd_sre_diarization/rttm

  # Create wav.scp file for diarization data (merge two telephone channels)
  python3 scripts/swbd_sre/create_wav_scp.py data/swbd_sre_filtered data/swbd_sre_diarization

  # Create utt2spk, spk2utt, utt2dur file
  awk -F' ' '{print $1" "$1}' data/swbd_sre_diarization/wav.scp > data/swbd_sre_diarization/utt2spk
  awk -F' ' '{print $1" "$1}' data/swbd_sre_diarization/wav.scp > data/swbd_sre_diarization/spk2utt
  utils/data/get_utt2dur.sh --nj 40 --cmd "$train_cmd" data/swbd_sre_diarization || exit 1;
  utils/fix_data_dir.sh data/swbd_sre_diarization || exit 1; # 31394 utterances (2 telephone channels merged)
fi

if [ $stage -le 4 ]; then
  # Filter out some "bad" audios (data cleaning)
  # Some single channel telephone speech is not clean enough. You can clearly hear two speakers talking. 
  # For example, swbd2-sw_13189-sw_1258-sw_1808 and swbdc-sw_41720-sw_5028-sw_5305
  mkdir -p data/swbd_sre_diarization_clean || exit 1;
  python3 scripts/swbd_sre/filter_bad_utt.py data/swbd_sre_diarization data/swbd_sre_diarization_clean 
  utils/filter_scp.pl data/swbd_sre_diarization_clean/wav.scp data/swbd_sre_diarization/utt2spk > data/swbd_sre_diarization_clean/utt2spk
  utils/filter_scp.pl data/swbd_sre_diarization_clean/wav.scp data/swbd_sre_diarization/spk2utt > data/swbd_sre_diarization_clean/spk2utt
  utils/filter_scp.pl data/swbd_sre_diarization_clean/wav.scp data/swbd_sre_diarization/utt2dur > data/swbd_sre_diarization_clean/utt2dur
  utils/filter_scp.pl -f 2 data/swbd_sre_diarization_clean/wav.scp data/swbd_sre_diarization/rttm > data/swbd_sre_diarization_clean/rttm
  # Create the mapping from speaker to index
  python3 scripts/create_spk2idx.py data/swbd_sre_diarization_clean
  utils/fix_data_dir.sh data/swbd_sre_diarization_clean || exit 1; # 29697 utterances (2 telephone channels merged), 2880.72 hours 
fi

uttdur=10.0
if [ $stage -le 5 ]; then
  # Split the utterances into short chunks (10 seconds)
  scripts/split_utt.sh --cmd "$train_cmd" --nj 40 --uttlen $uttdur --debug $debug \
	  data/swbd_sre_diarization_clean data/swbd_sre_diarization_clean_10s
  awk -F' ' -v dur="$uttdur" '{print $1, dur}' data/swbd_sre_diarization_clean_10s/wav.scp > data/swbd_sre_diarization_clean_10s/reco2dur
fi

if [ $stage -le 6 ]; then
  # Data augmentation with reverberation, noise, music (no babble)
  # We use the same data augmentation methods as Kaldi's speaker id example
  # The augmentation is based on MUSAN corpus (https://www.openslr.org/17)
  scripts/augmentation.sh --sample_rate 8000 --musan_dir /export/corpora/JHU/musan --prepare_musan true data/swbd_sre_diarization_clean_10s
  awk -F' ' -v dur="$uttdur" '{print $1, dur}' data/swbd_sre_diarization_clean_10s_combined/wav.scp > data/swbd_sre_diarization_clean_10s_combined/reco2dur
  # collect information for each 10s segment for training convenience
  scripts/create_record.sh --cmd "$train_cmd" --nj 40 data/swbd_sre_diarization_clean_10s_combined
fi

if [ $stage -le 7 ]; then
  # Split the dataset into train/swbd_dev/swbd_test. These three datasets have no speaker in common.
  python3 scripts/swbd_sre/split_train_dev_test.py --debug $debug data/swbd_sre_diarization_clean_10s_combined data/swbd_sre_final
  # Further split the train set into train_train and train_dev. The speakers in train_train contains train_dev
  # so that we can compute cross entropy loss on the dev set. 
  python3 scripts/swbd_sre/split_train_dev.py --debug $debug data/swbd_sre_final/train data/swbd_sre_final
  ln -s `pwd`/data/swbd_sre_diarization_clean_10s_combined/data data/swbd_sre_final/train_train/.
  ln -s `pwd`/data/swbd_sre_diarization_clean_10s_combined/data data/swbd_sre_final/train_dev/.

  # After this stage, there will be 5 folders under data/swbd_sre_final. They are train, swbd_dev, swbd_test, 
  # train_train, train_dev. swbd_dev and swbd_test are used for evaluation and they have no overlap speaker
  # with the train. train_train, train_dev are two parts of train. We use train_train for training and train_dev
  # for validation during training.
fi
