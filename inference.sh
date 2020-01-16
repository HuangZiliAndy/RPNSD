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

for fold_num in {1..5}; do
  test_dir=data/callhome_5folds/$fold_num/test

  exp_dir=$exp_dir_all/$fold_num 

  output_dir=$exp_dir/result/$modelname
  test_output_dir=$output_dir/callhome_test

  if [ $stage -le 0 ]; then
    echo "Modelname is $modelname"
    scripts/eval_cpu.sh --cmd "$train_cmd --mem 10G" --nj 40 \
            --nclass 1284 $test_dir $exp_dir $modelname callhome_test || exit 1;
  fi
  
  overlap=true
  if [ $stage -le 1 ]; then
    python3 scripts/cluster_nms.py $test_output_dir/detections.pkl $test_output_dir/rttm_overlap${overlap}_num_spk \
            --num_cluster $test_dir/reco2num_spk --nms_thres $nms_thres --thres $thres --cluster_type $cluster_type || exit 1;

    result_dir=$output_dir/results_thres${thres}_overlap${overlap}
    mkdir -p $result_dir
    cp $test_output_dir/rttm_overlap${overlap}_num_spk $result_dir/rttm_overlap${overlap}_num_spk_test || exit 1;

    for collar in 0 0.1 0.25; do
      if $overlap; then
        scoreopt="-c $collar"
      else
        scoreopt="-1 -c $collar"
      fi

      md-eval.pl $scoreopt -r $test_dir/rttm \
              -s $result_dir/rttm_overlap${overlap}_num_spk_test 2> $result_dir/collar${collar}_overlap${overlap}_num_spk_test.log \
              > $result_dir/collar${collar}_overlap${overlap}_DER_num_spk_test.txt
      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
        $result_dir/collar${collar}_overlap${overlap}_DER_num_spk_test.txt)
      echo "Fold $fold_num Oracle Collar $collar Overlap $overlap Callhome TEST DER: $der%"
    done
  fi

  overlap=false
  if [ $stage -le 2 ]; then
    python3 scripts/cluster_nms.py $test_output_dir/detections.pkl $test_output_dir/rttm_overlap${overlap}_num_spk \
            --num_cluster $test_dir/reco2num_spk --nms_thres $nms_thres --thres $thres --cluster_type $cluster_type || exit 1;

    result_dir=$output_dir/results_thres${thres}_overlap${overlap}
    mkdir -p $result_dir
    cp $test_output_dir/rttm_overlap${overlap}_num_spk $result_dir/rttm_overlap${overlap}_num_spk_test || exit 1;

    for collar in 0 0.1 0.25; do
      if $overlap; then
        scoreopt="-c $collar"
      else
        scoreopt="-1 -c $collar"
      fi

      md-eval.pl $scoreopt -r $test_dir/rttm \
              -s $result_dir/rttm_overlap${overlap}_num_spk_test 2> $result_dir/collar${collar}_overlap${overlap}_num_spk_test.log \
              > $result_dir/collar${collar}_overlap${overlap}_DER_num_spk_test.txt
      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
        $result_dir/collar${collar}_overlap${overlap}_DER_num_spk_test.txt)
      echo "Fold $fold_num Oracle Collar $collar Overlap $overlap Callhome TEST DER: $der%"
    done
  fi
done

if [ $stage -le 3 ]; then
  result_dir_all=$exp_dir_all/results
  mkdir -p $result_dir_all || exit 1;
  
  for overlap in "true" "false"; do
    for condition in "num_spk"; do
      cat $exp_dir_all/1/result/$modelname/results_thres${thres}_overlap${overlap}/rttm_overlap${overlap}_${condition}_test \
        $exp_dir_all/2/result/$modelname/results_thres${thres}_overlap${overlap}/rttm_overlap${overlap}_${condition}_test \
        $exp_dir_all/3/result/$modelname/results_thres${thres}_overlap${overlap}/rttm_overlap${overlap}_${condition}_test \
        $exp_dir_all/4/result/$modelname/results_thres${thres}_overlap${overlap}/rttm_overlap${overlap}_${condition}_test \
        $exp_dir_all/5/result/$modelname/results_thres${thres}_overlap${overlap}/rttm_overlap${overlap}_${condition}_test \
        > $result_dir_all/overlap${overlap}_callhome_${condition}_rttm

      for collar in 0 0.1 0.25; do
        if $overlap; then
          scoreopt="-c $collar"
        else
          scoreopt="-1 -c $collar"
        fi
        md-eval.pl $scoreopt -r $callhome_dir/rttm \
                -s $result_dir_all/overlap${overlap}_callhome_${condition}_rttm 2> $result_dir_all/collar${collar}_overlap${overlap}_${condition}.log \
                > $result_dir_all/collar${collar}_overlap${overlap}_DER_${condition}.txt
        der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
          $result_dir_all/collar${collar}_overlap${overlap}_DER_${condition}.txt)
        if [ $condition == "threshold" ]; then
          echo "Collar $collar Overlap $overlap Callhome DER: $der%"
        elif [ $condition == "num_spk" ]; then
          echo "Oracle Collar $collar Overlap $overlap Callhome DER: $der%"
        else
          echo "Condition not defined."
          exit 1;
        fi
      done
    done
  done
fi
