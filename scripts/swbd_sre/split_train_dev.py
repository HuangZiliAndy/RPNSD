#!/usr/bin/env python3

# This script splits the train set into train and dev
# parts (the speakers in the train set contain the speakers
# in the dev set, so that we can compute cross entropy loss
# on dev set)

import os
import sys
import random
import subprocess
import argparse

parser = argparse.ArgumentParser("Split train into train_train and train_dev sets")
parser.add_argument('data_dir', type=str, help='data directory')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--num_dev', type=int, default=2000, help='number of utterances for the dev set')
parser.add_argument('--seed', type=int, default=7, help='random seed')
parser.add_argument('--debug', type=int, default=0, help='debug mode')
parser.add_argument('--use_same_setting', type=int, default=1, help='I made a mistake in the first version \
        (I forget to sort the list before shuffling it) and I cannot get the exactly same train/dev \
        partition. If you want to use exactly the same configuration as the paper, set use_same_setting to 1')

def get_oriuttname(uttname):
    if uttname.endswith('music') or uttname.endswith('noise') or uttname.endswith('reverb') or uttname.endswith('babble'):
        uttname = '-'.join(uttname.split('-')[:-1])
    ori_uttname = '-'.join(uttname.split('-')[:-2])
    return ori_uttname

def get_spkdict(filename):
    with open(filename, 'r') as fh:
        content = fh.readlines()
    spk_dict = {}
    for line in content:
        line = line.strip('\n')
        uttname = line.split()[0]
        spk_ch1, spk_ch2 = get_spkname(uttname)
        if spk_ch1 not in spk_dict:
            spk_dict[spk_ch1] = 1
        if spk_ch2 not in spk_dict:
            spk_dict[spk_ch2] = 1
    return spk_dict

def get_spkname(uttname):
    if uttname.endswith("music") or uttname.endswith("noise") or uttname.endswith("reverb") or uttname.endswith("babble"):
        uttname = '-'.join(uttname.split('-')[:-1])
    spk_ch1, spk_ch2 = uttname.split('-')[-4], uttname.split('-')[-3]
    return spk_ch1, spk_ch2

def main(args):
    print("Split train into train_train and train_dev sets")
    random.seed(args.seed)

    with open("{}/wav.scp".format(args.data_dir), 'r') as fh:
        content = fh.readlines()
    uttdict = {}
    ori_uttname_dict = {}
    for line in content:
        line = line.strip('\n')
        uttname, ark = line.split()[0], " ".join(line.split()[1:])
        uttdict[uttname] = ark
        ori_uttname_dict[get_oriuttname(uttname)] = 1

    ori_uttlist = list(ori_uttname_dict.keys())
    ori_uttlist.sort()
    # choose dev utterances from SRE08, SRE10, SWBD (because SRE04, SRE05, SRE06 might share some utterances)
    select_uttlist = [utt for utt in ori_uttlist if "SRE08" in utt or "SRE10" in utt or "swbd" in utt]

    if args.debug:
        args.num_dev = int(len(ori_uttlist) * 0.1)

    if args.use_same_setting:
        import pickle
        with open("local/swbd_train_dev_utt.pkl", 'rb') as fh:
            ori_dev_uttlist = pickle.load(fh)
    else:
        random.shuffle(select_uttlist)
        ori_dev_uttlist = select_uttlist[:args.num_dev]
    ori_train_uttlist = [utt for utt in ori_uttlist if utt not in ori_dev_uttlist]
    print("{} train utts, {} dev utts".format(len(ori_train_uttlist), len(ori_dev_uttlist)))
    
    uttlist = list(uttdict.keys())
    uttlist.sort()
    dev_uttlist, train_uttlist = [], []
    ori_dev_uttdict, ori_train_uttdict = {}, {}
    for utt in ori_train_uttlist:
        ori_train_uttdict[utt] = 1
    for utt in ori_dev_uttlist:
        ori_dev_uttdict[utt] = 1

    for utt in uttlist:
        ori_uttname = get_oriuttname(utt)
        if ori_uttname in ori_dev_uttdict:
            dev_uttlist.append(utt)
        elif ori_uttname in ori_train_uttdict:
            train_uttlist.append(utt)
    print("total {} segments, {} train segments, {} dev segments".format(len(uttlist), len(train_uttlist), len(dev_uttlist)))

    train_dir, dev_dir = "{}/train_train".format(args.output_dir), "{}/train_dev".format(args.output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(dev_dir):
        os.makedirs(dev_dir)

    with open("{}/wav.scp".format(train_dir), 'w') as fh:
        for utt in train_uttlist:
            fh.write("{}\n".format(utt))
    train_spk_dict = get_spkdict("{}/wav.scp".format(train_dir))
    print("{} train spks".format(len(train_spk_dict)))

    with open("{}/wav.scp".format(dev_dir), 'w') as fh:
        for utt in dev_uttlist:
            spk_ch1, spk_ch2 = get_spkname(utt)
            if not args.debug:
                # make sure the speakers in the dev set are also in the training set
                if spk_ch1 in train_spk_dict and spk_ch2 in train_spk_dict:
                    fh.write("{}\n".format(utt))
            else:
                fh.write("{}\n".format(utt))
    print("")
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
