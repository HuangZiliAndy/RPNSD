#!/usr/bin/env python3

# Split train/dev/test, which correspond to
# Mixer 6 + SRE + SWBD, SWBD DEV, SWBD TEST
# in the paper. Since Mixer 6 and SRE have some
# complex data inclusion, we design experiments
# on SWBD. Note that, Mixer 6 + SRE + SWBD, 
# SWBD DEV, SWBD TEST have no speaker in common.

import os
import sys
import random
import subprocess
import argparse

parser = argparse.ArgumentParser("Split the whole dataset into train/dev/test sets (no speaker overlap)")
parser.add_argument('data_dir', type=str, help='data directory')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--num_dev_test', type=int, default=50, help='number of utterances to fetch for dev/test sets')
parser.add_argument('--seed', type=int, default=7, help='random seed')
parser.add_argument('--debug', type=int, default=0, help='debug mode')
parser.add_argument('--use_same_setting', type=int, default=1, help='I made a mistake in the first version \
        (I forget to sort the list before shuffling it) and I cannot get the exactly same train/dev/test \
        partition. If you want to use exactly the same configuration as the paper, set use_same_setting to 1')

def get_spkname(uttname):
    return [uttname.split('-')[-2], uttname.split('-')[-1]]

# We aim to select a small dataset D from the large dataset T
# The process is as follows
# (1) select num_utt utterances from the dataset.
# (2) get the speakers of these utterances (every utterance has 2 speakers)
# and we name it the speaker pool S.
# (3) consider every utterance u in the dataset T
# if both speakers of u are in the speaker pool S,
# append u in D.
# if only one speaker of u are in the speaker pool S,
# discard it.
# if neither speakers of u are in the speaker pool S,
# append u in another dataset R.

# After the process, we get datasets D and R. D and R have
# no common speaker. We return the speakers and utterances
# of D and utterances of R. 
def fetch_utt(utt_dict, num_utt, spk_list):
    uttlist = list(utt_dict.keys())
    uttlist.sort()
    random.shuffle(uttlist)
    utt_list_fetch = uttlist[:num_utt]

    if spk_list is None:
        spk_list = []
        for utt in utt_list_fetch:
            spk_list += get_spkname(utt)
        spk_list = list(set(spk_list))

    spk_dict = {spk:1 for spk in spk_list}
    new_utt_dict, utt_list_fetch = {}, []
    for utt in utt_dict.keys():
        spk1, spk2 = get_spkname(utt)
        if spk1 in spk_dict and spk2 in spk_dict:
            utt_list_fetch.append(utt)
        elif spk1 in spk_dict or spk2 in spk_dict:
            continue
        else:
            new_utt_dict[utt] = 1
    return spk_list, utt_list_fetch, new_utt_dict

def get_oriuttname(uttname):
    if uttname.endswith('music') or uttname.endswith('noise') or uttname.endswith('reverb') or uttname.endswith('babble'):
        uttname = '-'.join(uttname.split('-')[:-1])
    ori_uttname = '-'.join(uttname.split('-')[:-2])
    return ori_uttname

def check_every_utt_in_spklist(utt_list, spk_list):
    for utt in utt_list:
        spk1, spk2 = get_spkname(utt)
        assert spk1 in spk_list and spk2 in spk_list
    return 0

def check_swbd_train_dev_test(train_spk_list, dev_spk_list, test_spk_list, train_utt_list, dev_utt_list, test_utt_list):
    train_spk_set, dev_spk_set, test_spk_set = set(train_spk_list), set(dev_spk_list), set(test_spk_list)
    # make sure the speakers in train, dev and test have no overlap
    assert len(train_spk_set.intersection(dev_spk_set)) == 0
    assert len(train_spk_set.intersection(test_spk_set)) == 0
    assert len(dev_spk_set.intersection(test_spk_set)) == 0
    # make sure every speakers in the utterances are in the speaker list
    check_every_utt_in_spklist(train_utt_list, train_spk_list)
    check_every_utt_in_spklist(dev_utt_list, dev_spk_list)
    check_every_utt_in_spklist(test_utt_list, test_spk_list)
    return 0

def get_spklist(uttlist):
    spklist = []
    for utt in uttlist:
        spklist += get_spkname(utt)
    spklist = list(set(spklist))
    return spklist

def main(args):
    print("Split the whole dataset into train/dev/test sets (no speaker overlap)")
    if args.debug:
        args.num_dev_test = 5
    random.seed(args.seed)

    with open("{}/wav.scp".format(args.data_dir), 'r') as fh:
        content = fh.readlines()
    utt2ark = {}
    ori_uttname_dict, ori_uttname_dict_sre, ori_uttname_dict_swbd = {}, {}, {}
    for line in content:
        line = line.strip('\n')
        uttname, ark = line.split()[0], " ".join(line.split()[1:])
        utt2ark[uttname] = ark
        oriuttname = get_oriuttname(uttname)
        ori_uttname_dict[oriuttname] = 1
        if "swbd" in oriuttname:
            ori_uttname_dict_swbd[oriuttname] = 1
        else:
            ori_uttname_dict_sre[oriuttname] = 1
    spk_list_all, spk_list_swbd, spk_list_sre = get_spklist(list(ori_uttname_dict.keys())), get_spklist(list(ori_uttname_dict_swbd.keys())), get_spklist(list(ori_uttname_dict_sre.keys()))

    # count the number of utterances and speakers in SRE and SWBD
    print("In the whole dataset")
    print("total {} utts, {} swbd utts, {} sre utts".format(len(ori_uttname_dict), len(ori_uttname_dict_swbd), len(ori_uttname_dict_sre)))
    print("total {} spks, {} swbd spks, {} sre spks".format(len(spk_list_all), len(spk_list_swbd), len(spk_list_sre)))

    # split train, dev, test for SWBD
    if args.use_same_setting:
        import pickle
        with open("local/swbd_dev_spk.pkl", 'rb') as fh:
            dev_spk_list = pickle.load(fh)
        with open("local/swbd_test_spk.pkl", 'rb') as fh:
            test_spk_list = pickle.load(fh)
    else:
        dev_spk_list, test_spk_list = None, None
    dev_spk_list, dev_utt_list, new_ori_uttname_dict_swbd = fetch_utt(ori_uttname_dict_swbd, args.num_dev_test, dev_spk_list)
    test_spk_list, test_utt_list, new_ori_uttname_dict_swbd = fetch_utt(new_ori_uttname_dict_swbd, args.num_dev_test, test_spk_list)
    train_utt_list = list(new_ori_uttname_dict_swbd.keys())
    train_spk_list = get_spklist(train_utt_list)
    check_swbd_train_dev_test(train_spk_list, dev_spk_list, test_spk_list, train_utt_list, dev_utt_list, test_utt_list)

    # merge the utterances in SRE to train set
    train_utt_dict = {**new_ori_uttname_dict_swbd, **ori_uttname_dict_sre}
    dev_utt_dict = {utt:1 for utt in dev_utt_list}
    test_utt_dict = {utt:1 for utt in test_utt_list}
    spk_train, spk_dev, spk_test = get_spklist(list(train_utt_dict.keys())), get_spklist(list(dev_utt_dict.keys())), get_spklist(list(test_utt_dict.keys()))
    print("After splitting the whole dataset")
    print("{} train utts, {} dev utts, {} test utts".format(len(train_utt_dict), len(dev_utt_dict), len(test_utt_dict)))
    print("{} train spks, {} dev spks, {} test spks".format(len(spk_train), len(spk_dev), len(spk_test)))

    train_dir, dev_dir, test_dir = "{}/train".format(args.output_dir), "{}/swbd_dev".format(args.output_dir), "{}/swbd_test".format(args.output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(dev_dir):
        os.makedirs(dev_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    uttlist = list(utt2ark.keys()) 
    uttlist.sort()
    train_uttlist, dev_uttlist, test_uttlist = [], [], []
    for utt in uttlist:
        ori_uttname = get_oriuttname(utt)
        if ori_uttname in train_utt_dict:
            train_uttlist.append(utt)
        elif ori_uttname in dev_utt_dict:
            dev_uttlist.append(utt)
        elif ori_uttname in test_utt_dict:
            test_uttlist.append(utt)
    print("{} train segments, {} dev segments, {} test segments".format(len(train_uttlist), len(dev_uttlist), len(test_uttlist)))

    with open("{}/wav.scp".format(train_dir), 'w') as fh:
        for utt in train_uttlist:
            fh.write('{}\n'.format(utt))
    with open("{}/wav.scp".format(dev_dir), 'w') as fh:
        for utt in dev_uttlist:
            fh.write('{}\n'.format(utt))
    with open("{}/wav.scp".format(test_dir), 'w') as fh:
        for utt in test_uttlist:
            fh.write('{}\n'.format(utt))
    print("")
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
