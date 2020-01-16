#!/bin/bash
# Data augmentation with reverberation, noise, music and babble (optional).
# I didn't use babble augmentation

stage=0
use_babble=false
sample_rate=8000
prepare_musan=true
musan_dir=/export/corpora/JHU/musan

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 <data_dir>"
  echo "E.g.: $0 data/train"
  exit 1;
fi

srcdir=$1

if [ $stage -le 0 ]; then
  if [ ! -f $srcdir/reco2dur ]; then
    echo "Please prepare reco2dur file first"
    exit 1;
  fi

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate $sample_rate \
    $srcdir ${srcdir}_reverb
  utils/copy_data_dir.sh --utt-suffix "-reverb" ${srcdir}_reverb ${srcdir}_reverb.new
  rm -rf ${srcdir}_reverb
  mv ${srcdir}_reverb.new ${srcdir}_reverb

  # Fix the randomness problem in SoX
  mv ${srcdir}_reverb/wav.scp ${srcdir}_reverb/wav.scp.bak
  sed s/"sox "/"sox -R "/g ${srcdir}_reverb/wav.scp.bak > ${srcdir}_reverb/wav.scp

  if $prepare_musan; then
    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    steps/data/make_musan.sh --sampling-rate $sample_rate $musan_dir data

    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    for name in speech noise music; do
      mv data/musan_${name}/wav.scp data/musan_${name}/wav.scp.bak
      sed s/"sox "/"sox -R "/g data/musan_${name}/wav.scp.bak > data/musan_${name}/wav.scp
      utils/data/get_utt2dur.sh data/musan_${name}
      mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    done
  fi

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" ${srcdir} ${srcdir}_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" ${srcdir} ${srcdir}_music
  if $use_babble; then
    # Augment with musan_speech
    steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" ${srcdir} ${srcdir}_babble
    # Combine reverb, noise, music, and babble into one directory.
    utils/combine_data.sh ${srcdir}_aug ${srcdir}_reverb ${srcdir}_noise ${srcdir}_music ${srcdir}_babble
  else
    # Combine reverb, noise, music into one directory.
    utils/combine_data.sh ${srcdir}_aug ${srcdir}_reverb ${srcdir}_noise ${srcdir}_music
  fi

  utils/combine_data.sh ${srcdir}_combined ${srcdir}_aug ${srcdir}
fi

if [ $stage -le 1 ]; then
  # Create label.scp for augmented data directory
  python scripts/prepare_label_scp_aug.py ${srcdir} ${srcdir}_combined 
fi
