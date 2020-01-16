#!/usr/bin/env python3

# This script is called by scripts/create_record.sh
# The dataset is too large after data augmentation, even
# the mapping from utterance to feature consumes a lot of
# memory. So this script records the necessary information
# for each small file.

import os
import sys
import argparse

parser = argparse.ArgumentParser("Create information for each segment")
parser.add_argument('data_dir', type=str, help='data directory')
parser.add_argument('output_dir', type=str, help='output directory')

def load_wav_scp(wav_scp_file):
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
    return {x[0]: x[1] for x in lines}

def load_reco2dur(reco2dur_file):
    lines = [line.strip().split(None, 1) for line in open(reco2dur_file)]
    return {x[0]: float(x[1]) for x in lines}

def load_label_scp(label_scp_file):
    lines = [line.strip().split(None, 1) for line in open(label_scp_file)]
    return {x[0]: x[1] for x in lines}

def main(args):
    rec2dur = load_reco2dur("{}/reco2dur".format(args.data_dir))
    utt2wav = load_wav_scp("{}/wav.scp".format(args.data_dir))
    utt2label = load_label_scp("{}/label.scp".format(args.data_dir))
    assert len(rec2dur) == len(utt2wav)
    uttlist = list(utt2wav.keys())
    cnt = 0
    for utt in uttlist:
        with open("{}/{}.txt".format(args.output_dir, utt), 'w') as fh:
            fh.write("{} {} {} {}\n".format(utt, rec2dur[utt], utt2label[utt], utt2wav[utt]))
        cnt += 1
        if cnt % 1000 == 0:
            print("finish {}/{}".format(cnt, len(uttlist)), flush=True)
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
