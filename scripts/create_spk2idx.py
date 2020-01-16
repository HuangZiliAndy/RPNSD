#!/usr/bin/env python3

# This script creates spk2idx file from RTTM file

import os
import sys
import argparse 

parser = argparse.ArgumentParser("Create spk2idx")
parser.add_argument('data_dir', type=str, help='data directory')

def get_spklist(rttm_file):
    spklist = []
    with open(rttm_file, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        spklist.append(line_split[7])
    spklist = list(set(spklist))
    spklist.sort()
    return spklist

def write_spk2idx(spklist, spk2idx_file):
    with open(spk2idx_file, 'w') as fh:
        for i in range(len(spklist)):
            fh.write("{} {}\n".format(spklist[i], i))
    return 0

def main(args):
    spklist = get_spklist("{}/rttm".format(args.data_dir))
    write_spk2idx(spklist, "{}/spk2idx".format(args.data_dir))
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
