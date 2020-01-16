#!/usr/bin/env python3

# This script creates wav.scp file for diarization data (merge two telephone channels)
# This script is only for SWBD/SRE

import os
import sys
import argparse 

parser = argparse.ArgumentParser("Create wav.scp")
parser.add_argument('src_dir', type=str, help='source directory (single channel telephone conversation)')
parser.add_argument('tgt_dir', type=str, help='target directory (two telephone channels merged)')

def create_wav_scp(src_dir, tgt_dir):
    utt_dict = prepare_utt_dict("{}/rttm".format(tgt_dir))

    wav_scp_file = open("{}/wav.scp".format(tgt_dir), 'w')
    with open("{}/wav.scp".format(src_dir), 'r') as fh:
        content = fh.readlines()

    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname = line_split[0]
        ori_uttname = uttname[:-2]
        if ori_uttname not in utt_dict:
            continue
        ark_info = " ".join(line_split[1:])
        if uttname[-1] == "1":
            pos = ark_info.find("-c 1 ")
            assert pos != -1
            ark_info = ark_info.replace("-c 1 ", "") # remove the channel information
            wav_scp_file.write("{} {}\n".format(uttname[:-2], ark_info + " sox -R - -r 8000 -c 1 -t wav - remix 1,2 |"))
        elif uttname[-1] == "2":
            continue
        else:
            raise ValueError("Condition not defined.")
    wav_scp_file.close()
    return 0

def prepare_utt_dict(rttm_file):
    utt_dict = {}
    with open(rttm_file, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        uttname = line.split()[1]
        utt_dict[uttname] = 1
    print("{} utterances in RTTM file".format(len(utt_dict)))
    return utt_dict

def main(args):
    # create wav.scp file
    create_wav_scp(args.src_dir, args.tgt_dir)
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
