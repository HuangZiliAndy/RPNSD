#!/usr/bin/env python3

# This script creates Callhome data directory with whole utterances

import os
import sys
import argparse

def get_oriuttname(uttname):
    if uttname.endswith('music') or uttname.endswith('noise') or uttname.endswith('reverb') or uttname.endswith('babble'):
        uttname = '-'.join(uttname.split('-')[:-1])
    ori_uttname = '-'.join(uttname.split('-')[:-2])
    return ori_uttname

def load_wav_scp(filename):
    utt2ark = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        uttname, ark = line.split(None, 1)
        utt2ark[uttname] = ark
    return utt2ark

def get_uttlist(data_dir):
    uttlist = []
    with open("{}/wav.scp".format(data_dir), 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        uttname = line.split()[0]
        ori_uttname = get_oriuttname(uttname)
        uttlist.append(ori_uttname)
    uttlist = list(set(uttlist))
    uttlist.sort()
    return uttlist

def get_args():
    parser = argparse.ArgumentParser(
      description="""This script creates Callhome data 
      directory with whole utterances""")

    parser.add_argument("src_dir", type=str,
                        help="source data directory")
    parser.add_argument("tgt_dir", type=str,
                        help="target data directory")
    parser.add_argument("callhome_dir", type=str,
                        help="callhome data directory")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="number of folds")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    utt2ark = load_wav_scp("{}/wav.scp".format(args.callhome_dir)) 
    for i in range(args.num_folds):
        fold_num = i + 1
        uttlist_dict = {}
        for condition in ["train", "dev", "test"]:
            data_dir = "{}/{}/{}".format(args.src_dir, fold_num, condition)
            assert os.path.exists(data_dir)
            uttlist = get_uttlist(data_dir) 
            uttlist_dict[condition] = uttlist

        for condition in ["train", "dev", "test"]:
            data_dir = "{}/{}/{}".format(args.tgt_dir, fold_num, condition)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            uttlist = uttlist_dict[condition]
            with open("{}/wav.scp".format(data_dir), 'w') as fh:
                for utt in uttlist:
                    fh.write("{} {}\n".format(utt, utt2ark[utt]))
    return 0

if __name__ == "__main__":
    main()
