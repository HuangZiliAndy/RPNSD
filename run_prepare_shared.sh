#!/bin/bash

# This script prepares the datasets and some necessary libraries

stage=0
. path.sh

if [ $stage -le 0 ]; then
  # Prepare SWBD/SRE dataset
  # In this script, I am using the data path on JHU grid.
  # Please change that to your own path before use that
  ./scripts/swbd_sre/prepare_swbd_sre.sh
  # Prepare CALLHOME dataset
  # The CALLHOME dataset doesn't specify train/dev/test
  # Therefore we use 5-fold cross validation
  ./scripts/callhome/prepare_callhome_5folds.sh
fi
