#!/usr/bin/env python3

# This script split the Callhome dataset into folds

import os
import sys
import argparse
import random

def get_args():
    parser = argparse.ArgumentParser(
      description="""This script split the Callhome dataset into folds""")
    parser.add_argument("src_dir", type=str,
                        help="Source data directory")
    parser.add_argument("tgt_dir", type=str,
                        help="Target data directory")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="Number of folds")
    parser.add_argument("--dev_portion", type=float, default=0.1,
                        help="Portion of dev set")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed")
    args = parser.parse_args()
    return args

def get_oriuttname(uttname):
    if uttname.endswith('music') or uttname.endswith('noise') or uttname.endswith('reverb') or uttname.endswith('babble'):
        uttname = '-'.join(uttname.split('-')[:-1])
    ori_uttname = '-'.join(uttname.split('-')[:-2])
    return ori_uttname

def load_wav_scp(filename):
    utt2ark, utt2subutt = {}, {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        uttname, ark = line.split(None, 1)
        utt2ark[uttname] = ark
        oriuttname = get_oriuttname(uttname)
        if oriuttname not in utt2subutt:
            utt2subutt[oriuttname] = []
        utt2subutt[oriuttname].append(uttname)
    return utt2ark, utt2subutt

def write_wav_scp(data_dir, uttlist, utt2ark, utt2subutt):
    with open("{}/wav.scp".format(data_dir), 'w') as fh:
        for utt in uttlist:
            subutt_list = utt2subutt[utt]
            for subutt in subutt_list:
                fh.write("{}\n".format(subutt))
    return 0

def main(args):
    random.seed(args.seed)
    utt2ark, utt2subutt = load_wav_scp("{}/wav.scp".format(args.src_dir))
    uttlist = list(set([get_oriuttname(utt) for utt in utt2ark]))
    uttlist.sort()
    assert len(utt2subutt) == len(uttlist)
    print("{} utts in total".format(len(uttlist)))

    random.shuffle(uttlist)
    num_utt_per_fold = int(len(uttlist) / args.num_folds) + 1
    utt_fold_list = []
    for i in range(args.num_folds):
        start_pos = num_utt_per_fold * i
        if i == args.num_folds - 1:
            end_pos = len(uttlist)
        else:
            end_pos = num_utt_per_fold * (i + 1)
        utt_fold = uttlist[start_pos : end_pos]
        utt_fold_list.append(utt_fold)

    for i in range(args.num_folds):
        test_utt = utt_fold_list[i]
        train_dev_utt = [utt for utt in uttlist if utt not in test_utt]
        num_dev = int(args.dev_portion * len(train_dev_utt))
        random.shuffle(train_dev_utt)
        train_utt = train_dev_utt[num_dev:]
        dev_utt = train_dev_utt[:num_dev]
        assert len(test_utt) + len(train_utt) + len(dev_utt) == len(uttlist)
        assert len(set(train_utt).intersection(set(dev_utt))) == 0
        assert len(set(train_utt).intersection(set(test_utt))) == 0
        assert len(set(dev_utt).intersection(set(test_utt))) == 0
        train_utt.sort()
        dev_utt.sort()
        test_utt.sort()
        print("Fold {}, train utts {}, dev utts {}, test utts {}".format(i + 1, len(train_utt), len(dev_utt), len(test_utt)))
        train_dir, dev_dir, test_dir = "{}/{}/train".format(args.tgt_dir, i + 1), "{}/{}/dev".format(args.tgt_dir, i + 1), "{}/{}/test".format(args.tgt_dir, i + 1)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(dev_dir):
            os.makedirs(dev_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        write_wav_scp(train_dir, train_utt, utt2ark, utt2subutt)
        write_wav_scp(dev_dir, dev_utt, utt2ark, utt2subutt)
        write_wav_scp(test_dir, test_utt, utt2ark, utt2subutt)
    return 0

if __name__ == "__main__":
    args = get_args()
    main(args)
