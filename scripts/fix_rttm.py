#!/usr/bin/env python3
# This script fixes some problems the RTTM file
# including invalid time boundaries and others

import os
import sys
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(
      description="Fix RTTM file")
    parser.add_argument("rttm_file", type=str,
                        help="Input RTTM file")
    parser.add_argument("rttm_output_file", type=str,
                        help="Output RTTM file")
    parser.add_argument("--channel", type=int, default=1,
                        help="Channel information in the RTTM file")
    parser.add_argument("--add_uttname", type=int, default=0,
                        help="Whether to add uttname to spkname")
    args = parser.parse_args()
    return args

def load_rttm(filename):
    utt2seg = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname, start_t, duration, spkname = line_split[1], float(line_split[3]), float(line_split[4]), line_split[7]
        if duration <= 0:
            print("Invalid line")
            print(line)
            continue
        end_t = start_t + duration
        if uttname not in utt2seg:
            utt2seg[uttname] = []
        utt2seg[uttname].append([start_t, end_t, spkname])
    return utt2seg

def merge_same_spk(seg_array):
    spk_list = list(set(seg_array[:, 2]))
    seg_array_list = []
    for spk in spk_list:
        seg_array_spk = seg_array[seg_array[:, 2] == spk]
        seg_list_spk = []
        for i in range(len(seg_array_spk)):
            if i == 0:
                seg_list_spk.append(seg_array_spk[i, :])
            else:
                if seg_array_spk[i, 0] > seg_list_spk[-1][1]:
                    seg_list_spk.append(seg_array_spk[i, :])
                else:
                    seg_list_spk[-1][1] = max(seg_list_spk[-1][1], seg_array_spk[i, 1])
        seg_array_spk_new = np.array(seg_list_spk)
        seg_array_list.append(seg_array_spk_new)
    seg_array_new = np.concatenate(seg_array_list)
    seg_array_new = seg_array_new[seg_array_new[:, 0].argsort(), :]
    return seg_array_new 

def fix_rttm(utt2seg):
    uttlist = list(utt2seg.keys())
    uttlist.sort()
    utt2seg_new = {}
    for utt in uttlist:
        seg_list = utt2seg[utt]
        spk_list = list(set([seg[2] for seg in seg_list]))
        spk_list.sort()
        seg_array = np.array([[seg[0], seg[1], spk_list.index(seg[2])] for seg in seg_list])
        seg_array = seg_array[seg_array[:, 0].argsort(), :]
        seg_array_new = merge_same_spk(seg_array)
        seg_list = []
        for i in range(len(seg_array_new)):
            seg_list.append([seg_array_new[i, 0], seg_array_new[i, 1], spk_list[int(seg_array_new[i, 2])]])
        utt2seg_new[utt] = seg_list
    return utt2seg_new

def write_rttm(utt2seg, rttm_output_file, add_uttname, channel):
    uttlist = list(utt2seg.keys())
    uttlist.sort()
    with open(rttm_output_file, 'w') as fh:
        for utt in uttlist:
            seg_list = utt2seg[utt]
            for seg in seg_list:
                if add_uttname:
                    fh.write("SPEAKER {} {} {:.2f} {:.2f} <NA> <NA> {}_{} <NA> <NA>\n".format(utt, channel, seg[0], seg[1] - seg[0], utt, seg[2]))
                else:
                    fh.write("SPEAKER {} {} {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(utt, channel, seg[0], seg[1] - seg[0], seg[2]))
    return 0

def main():
    args = get_args()
    # load input RTTM
    utt2seg = load_rttm(args.rttm_file)
    # fix RTTM file
    utt2seg_new = fix_rttm(utt2seg)
    # write output RTTM
    write_rttm(utt2seg_new, args.rttm_output_file, args.add_uttname, args.channel)
    return 0

if __name__ == "__main__":
    main()
