#!/usr/bin/env python3

# This script divides the train_dev set into new train/dev splits
# (train_dev_train and train_dev_dev), so that we can compute 
# speaker cross entropy loss on validation set

import os
import sys
import random
import subprocess
import argparse

parser = argparse.ArgumentParser("Split train/dev partition")
parser.add_argument('data_dir', type=str, help='data directory')
parser.add_argument('--dev_portion', type=float, default=0.1, help='dev portion')
parser.add_argument('--seed', type=int, default=7, help='random seed')

def get_segname(uttname):
    if uttname.endswith('music') or uttname.endswith('noise') or uttname.endswith('reverb') or uttname.endswith('babble'):
        segname = '-'.join(uttname.split('-')[:-1])
    else:
        segname = uttname
    return segname

def get_uttname(uttname):
    if uttname.endswith('music') or uttname.endswith('noise') or uttname.endswith('reverb') or uttname.endswith('babble'):
        uttname = '-'.join(uttname.split('-')[:-1])
    uttname = '-'.join(uttname.split('-')[:-2])
    return uttname

def load_rttm(filename):
    utt2spklist = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        uttname, spkname = line.split()[1], line.split()[7] 
        if uttname not in utt2spklist:
            utt2spklist[uttname] = []
        utt2spklist[uttname].append(spkname)
    for utt in utt2spklist:
        utt2spklist[utt] = list(set(utt2spklist[utt]))
        assert len(utt2spklist[utt]) == 2
    return utt2spklist

# For iatd-5000-6000-music, segaug is iatd-5000-6000-music (segment after augmentation)
# seg is iatd-5000-6000 (segment), utt is iatd (whole utterance)
def process_wav_scp(filename):
    segaugdict, segdict, uttdict = {}, {}, {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        segaugname = line.strip('\n')
        segaugdict[segaugname] = 1
        segname = get_segname(segaugname)
        segdict[segname] = 1
        uttname = get_uttname(segaugname)
        uttdict[uttname] = 1
    return segaugdict, segdict, uttdict

def main(args):
    for fold_num in range(1, 6):
        print("-" * 80)
        print("Fold number is {}".format(fold_num))
        data_fold_dir = "{}/{}".format(args.data_dir, fold_num)
        train_dir, dev_dir, test_dir = "{}/train".format(data_fold_dir), "{}/dev".format(data_fold_dir), "{}/test".format(data_fold_dir)
        segaugdict_train, segdict_train, uttdict_train = process_wav_scp("{}/wav.scp".format(train_dir))
        segaugdict_dev, segdict_dev, uttdict_dev = process_wav_scp("{}/wav.scp".format(dev_dir))
        segaugdict_test, segdict_test, uttdict_test = process_wav_scp("{}/wav.scp".format(test_dir))
        print("{} train utts, {} dev utts, {} test utts".format(len(uttdict_train), len(uttdict_dev), len(uttdict_test)))
        print("{} train segments, {} dev segments, {} test segments".format(len(segdict_train), len(segdict_dev), len(segdict_test)))
        print("{} train aug segments, {} dev aug segments, {} test aug segments".format(len(segaugdict_train), len(segaugdict_dev), len(segaugdict_test)))

        # split train and dev set
        segaugdict_train_dev = {**segaugdict_train, **segaugdict_dev}
        train_dev_seg_list = list(segdict_train.keys()) + list(segdict_dev.keys())
        train_dev_seg_list.sort()
        random.shuffle(train_dev_seg_list)

        num_dev = int(len(train_dev_seg_list) * args.dev_portion)
        train_seglist = train_dev_seg_list[num_dev:]
        dev_seglist = train_dev_seg_list[:num_dev]
        train_seglist.sort()
        dev_seglist.sort()

        print("New train/dev split")
        print("{} train segments, {} dev segments".format(len(train_seglist), len(dev_seglist)))

        # write wav.scp file
        new_train_dir, new_dev_dir = "{}/train_dev_train".format(data_fold_dir), "{}/train_dev_dev".format(data_fold_dir)
        if not os.path.exists(new_train_dir):
            os.makedirs(new_train_dir)
        if not os.path.exists(new_dev_dir):
            os.makedirs(new_dev_dir)
        with open("{}/wav.scp".format(new_train_dir), 'w') as fh:
            for seg in train_seglist:
                for condition in ["", "music", "noise", "reverb"]:
                    if condition == "":
                        segname = seg
                    else:
                        segname = "{}-{}".format(seg, condition)
                    assert segname in segaugdict_train_dev
                    fh.write("{}\n".format(segname))
        with open("{}/wav.scp".format(new_dev_dir), 'w') as fh:
            for seg in dev_seglist:
                for condition in ["", "music", "noise", "reverb"]:
                    if condition == "":
                        segname = seg
                    else:
                        segname = "{}-{}".format(seg, condition)
                    assert segname in segaugdict_train_dev
                    fh.write("{}\n".format(segname))
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
