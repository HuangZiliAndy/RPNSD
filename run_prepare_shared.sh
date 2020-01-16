#!/bin/bash

# This script prepares the datasets and some necessary libraries

stage=0
. path.sh

if [ $stage -le 0 ]; then
  # Clone and install https://github.com/jwyang/faster-rcnn.pytorch.git
  mkdir -p tools
  git clone https://github.com/jwyang/faster-rcnn.pytorch.git tools/faster-rcnn.pytorch
  cd tools/faster-rcnn.pytorch && mkdir data || exit 1;
  pip install -r requirements.txt || exit 1;
  cd lib
  sh make.sh || exit 1;
  cd ../../..
  ln -s `pwd`/tools/faster-rcnn.pytorch/lib/model/nms scripts/model/. 
  ln -s `pwd`/tools/faster-rcnn.pytorch/lib/model/roi_align scripts/model/. 
  ln -s `pwd`/tools/faster-rcnn.pytorch/lib/model/roi_crop scripts/model/. 
  ln -s `pwd`/tools/faster-rcnn.pytorch/lib/model/roi_pooling scripts/model/. 
fi

if [ $stage -le 1 ]; then
  # Prepare SWBD/SRE dataset
  # In this script, I am using the data path on JHU grid.
  # Please change that to your own path before use that
  ./scripts/swbd_sre/prepare_swbd_sre.sh
  # Prepare CALLHOME dataset
  # The CALLHOME dataset doesn't specify train/dev/test
  # Therefore we use 5-fold cross validation
  ./scripts/callhome/prepare_callhome_5folds.sh
fi
