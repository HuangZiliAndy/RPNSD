#!/usr/bin/env python3

# This script filters out the "bad" audios.
# Some single channel telephone speech is not clean enough. 
# You can clearly hear two speakers talking. 
# Listen to swbd2-sw_13189-sw_1258-sw_1808 and 
# swbdc-sw_41720-sw_5028-sw_5305 and you will
# understand where the problem is.

import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser("Filter out bad audios in SRE/SWBD dataset") 
parser.add_argument('src_dir', type=str, help='source directory')
parser.add_argument('tgt_dir', type=str, help='target directory')
parser.add_argument('--thres', type=float, default=0.3, help='decision threshold, the audios \
                    whose spk1 proportion and spk2 proportion are smaller than \
                    threshold are kept')

def process_utt2dur(filename):
    utt2dur = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2dur[line_split[0]] = float(line_split[1])
    return utt2dur

def process_rttm(filename):
    utt2seg = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname, start_time, duration, spk = line_split[1], float(line_split[3]), float(line_split[4]), line_split[7]
        end_time = start_time + duration
        if uttname not in utt2seg:
            utt2seg[uttname] = []
        utt2seg[uttname].append([start_time, end_time, spk])
    return utt2seg

# compute statistics for audios
def stat_info(utt2dur, utt2seg):
    uttlist = list(utt2dur.keys())
    uttlist.sort()
    utt2info = {}
    for utt in uttlist:
        duration, seg_list = utt2dur[utt], utt2seg[utt]
        spk_list = list(set([seg[2] for seg in seg_list]))
        if len(spk_list) == 1:
            print("{} only has one speaker, skipping it".format(utt))
            continue

        spk1_seg_list = [seg for seg in seg_list if seg[2] == spk_list[0]]
        spk2_seg_list = [seg for seg in seg_list if seg[2] == spk_list[1]]

        num_frames = int(round(duration * 100))
        spk1_label, spk2_label = np.zeros((num_frames, )), np.zeros((num_frames, ))

        for seg in spk1_seg_list:
            start_frame, end_frame = int(round(seg[0] * 100)), int(round(seg[1] * 100))
            assert end_frame <= num_frames and start_frame >= 0
            spk1_label[start_frame:end_frame] = 1

        for seg in spk2_seg_list:
            start_frame, end_frame = int(round(seg[0] * 100)), int(round(seg[1] * 100))
            assert end_frame <= num_frames and start_frame >= 0
            spk2_label[start_frame:end_frame] = 1

        intersect = np.sum(np.logical_and(spk1_label, spk2_label))
        union = np.sum(np.logical_or(spk1_label, spk2_label))
        utt2info[utt] = [1.0 * intersect / int(np.sum(spk1_label)), 1.0 * intersect / int(np.sum(spk2_label))]
    print("Successfully compute statistics for {}/{} audios".format(len(utt2info), len(uttlist)))
    return utt2info

def remove_utt(utt2info, thres):
    utt2info_keep = {}
    for utt in utt2info:
        if round(np.max(utt2info[utt]), 4) < thres:
            utt2info_keep[utt] = utt2info[utt]
    print("Keep {}/{} audios".format(len(utt2info_keep), len(utt2info)))
    return utt2info_keep

# write new wav.scp file
def write_wav_scp(src_dir, tgt_dir, utt2info_keep):
    with open('{}/wav.scp'.format(src_dir), 'r') as fh:
        content = fh.readlines()
    with open('{}/wav.scp'.format(tgt_dir), 'w') as fh:
        for line in content:
            uttname = line.split()[0]
            if uttname in utt2info_keep:
                fh.write(line)
    return 0

def main(args):
    utt2dur = process_utt2dur("{}/utt2dur".format(args.src_dir))
    utt2seg = process_rttm("{}/rttm".format(args.src_dir))
    utt2info = stat_info(utt2dur, utt2seg)
    utt2info_keep = remove_utt(utt2info, args.thres)
    write_wav_scp(args.src_dir, args.tgt_dir, utt2info_keep)
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
