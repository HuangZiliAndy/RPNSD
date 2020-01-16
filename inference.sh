#!/bin/bash

# inference steps for RPNSD

. ./cmd.sh
. ./path.sh

stage=0
callhome_dir=data/callhome

exp_dir_all=experiment/pretraincfgres101epoch1bs8opsgdlr0.01min_lr0.0001schedulermultipat10seed7alpha1.0archres101ls0dev12000modelbestfreeze0bnfix0cfgres101epoch10bs8opsgdlr0.00004min_lr0.00004pat10seed7alpha0.1archres101ls0
thres=0.5
nms_thres=0.3
cluster_type=kmeans
modelname=modelbest

echo "Experiment directory is $exp_dir_all"
echo "Decision threshold is $thres"
echo "NMS threshold is $nms_thres"

# Evaluate on the CALLHOME dataset
for fold_num in {1..5}; do
  test_dir=data/callhome_5folds/$fold_num/test
  exp_dir=$exp_dir_all/$fold_num 
  output_dir=$exp_dir/result/$modelname
  test_output_dir=$output_dir/callhome_test

  # Forward the model to get region proposals, confidence score and speaker embeddings 
  if [ $stage -le 0 ]; then
    echo "Fold $fold_num Modelname is $modelname"
    scripts/eval_cpu.sh --cmd "$train_cmd --mem 10G" --nj 40 \
            --nclass 1284 $test_dir $exp_dir $modelname callhome_test || exit 1;
  fi
  
  # Cluster the speaker embeddings and apply NMS
  if [ $stage -le 1 ]; then
    python3 scripts/cluster_nms.py $test_output_dir/detections.pkl $test_output_dir/rttm_num_spk \
            --num_cluster $test_dir/reco2num_spk --nms_thres $nms_thres --thres $thres --cluster_type $cluster_type || exit 1;
  fi

  # Compute DER
  if [ $stage -le 2 ]; then
    for overlap in true false; do
      for collar in 0 0.1 0.25; do
        if $overlap; then
          scoreopt="-c $collar"
        else
          scoreopt="-1 -c $collar"
        fi

        md-eval.pl $scoreopt -r $test_dir/rttm \
                -s $test_output_dir/rttm_num_spk 2> $test_output_dir/collar${collar}_overlap${overlap}_num_spk.log \
                > $test_output_dir/collar${collar}_overlap${overlap}_DER_num_spk.txt
        der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
          $test_output_dir/collar${collar}_overlap${overlap}_DER_num_spk.txt)
        echo "Oracle Collar $collar Overlap $overlap Callhome TEST DER: $der%"
      done
    done
  fi
done

# Concatenate the RTTM files for different folds and compute the DER
if [ $stage -le 3 ]; then
  result_dir_all=$exp_dir_all/results
  mkdir -p $result_dir_all || exit 1;
  
  cat $exp_dir_all/1/result/$modelname/callhome_test/rttm_num_spk \
    $exp_dir_all/2/result/$modelname/callhome_test/rttm_num_spk \
    $exp_dir_all/3/result/$modelname/callhome_test/rttm_num_spk \
    $exp_dir_all/4/result/$modelname/callhome_test/rttm_num_spk \
    $exp_dir_all/5/result/$modelname/callhome_test/rttm_num_spk \
    > $result_dir_all/rttm_num_spk

  for overlap in "true" "false"; do
    for collar in 0 0.1 0.25; do
      if $overlap; then
        scoreopt="-c $collar"
      else
        scoreopt="-1 -c $collar"
      fi
      md-eval.pl $scoreopt -r $callhome_dir/rttm \
              -s $result_dir_all/rttm_num_spk 2> $result_dir_all/collar${collar}_overlap${overlap}.log \
              > $result_dir_all/collar${collar}_overlap${overlap}_DER.txt
      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
        $result_dir_all/collar${collar}_overlap${overlap}_DER.txt)
      echo "Oracle Collar $collar Overlap $overlap Callhome DER: $der%"
    done
  done
fi
