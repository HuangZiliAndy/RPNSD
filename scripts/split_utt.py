#!/usr/bin/env python3

# This script splits long utterances into short segments
# making it more convenient for data loading

import os
import sys
import numpy as np
import subprocess
import argparse

parser = argparse.ArgumentParser("Splits long utterances into short segments")
parser.add_argument('src_dir', type=str, help='source directory')
parser.add_argument('tgt_dir', type=str, help='target directory')
parser.add_argument('--uttlen', type=float, default=10.0, help='utterance length')
parser.add_argument('--sample_rate', type=int, default=8000, help='utterance sample rate')
parser.add_argument('--debug', type=int, default=0, help='debug mode')

def process_wavscp(filename):
    utt2ark = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2ark[line_split[0]] = " ".join(line_split[1:])
    return utt2ark

def process_utt2dur(filename):
    utt2dur = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        utt2dur[line_split[0]] = float(line_split[1])
    return utt2dur

def process_spk2idx(filename):
    spk2idx = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        spk2idx[line_split[0]] = int(line_split[1])
    return spk2idx

def process_rttm(filename):
    utt2seg_list, utt2spk_list = {}, {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname, start_time, duration, spk = line_split[1], float(line_split[3]), float(line_split[4]), line_split[7]
        end_time = start_time + duration
        if uttname not in utt2seg_list:
            utt2seg_list[uttname] = []
        if uttname not in utt2spk_list:
            utt2spk_list[uttname] = []
        utt2seg_list[uttname].append([start_time, end_time])
        utt2spk_list[uttname].append([spk])
    return utt2seg_list, utt2spk_list

def main(args):
    utt2ark = process_wavscp("{}/wav.scp".format(args.src_dir))
    utt2dur = process_utt2dur("{}/utt2dur".format(args.src_dir))
    utt2seg_list, utt2spk_list = process_rttm("{}/rttm".format(args.src_dir))
    spk2idx = process_spk2idx("{}/spk2idx".format(args.src_dir))

    uttlist = list(utt2ark.keys())
    uttlist.sort()

    cnt = 0
    for utt in uttlist:
        duration = utt2dur[utt]
        ark = utt2ark[utt]
        seg_list, spk_list = utt2seg_list[utt], utt2spk_list[utt]
        seg_array, spk_array = np.array(seg_list), np.array(spk_list)

        # copy the audios into wav format
        if ark.endswith("|"):
            wav_filename = "{}/tmp/{}.wav".format(args.tgt_dir, utt)
            cmd = "{} sox -R -t wav - -r {} -t wav {}".format(ark, args.sample_rate, wav_filename)
            status, result = subprocess.getstatusoutput(cmd)
            assert status == 0
        elif ark.endswith(".wav"):
            wav_filename = ark
        else:
            raise ValueError("Condition not defined.")

        num_sub_utts = int(np.ceil(duration / args.uttlen))
        for i in range(num_sub_utts):
            if i == num_sub_utts - 1:
                end_t = int(duration)
                start_t = end_t - args.uttlen
            else:
                start_t = args.uttlen * i
                end_t = args.uttlen * (i + 1)

            # create sub wav file
            sub_filename = "{}-{}-{}".format(utt, int(round(start_t * 100)), int(round(end_t * 100)))
            sub_wav_filename = "{}/wav/{}.wav".format(args.tgt_dir, sub_filename)
            cmd = "sox -R -t wav {} -t wav {} trim {} {}".format(wav_filename, sub_wav_filename, start_t, args.uttlen)
            status, result = subprocess.getstatusoutput(cmd)
            assert status == 0

            # create sub label file
            idx = np.logical_and(seg_array[:, 0] < end_t, seg_array[:, 1] > start_t)
            sub_seg_array = seg_array[idx, :]
            sub_seg_array[sub_seg_array < start_t] = start_t
            sub_seg_array[sub_seg_array > end_t] = end_t
            sub_seg_array = sub_seg_array - start_t

            sub_spk_array = spk_array[idx, :]
            sub_spk_list = [spk2idx[spk[0]] for spk in list(sub_spk_array)]
            sub_spk_array = np.array(sub_spk_list)
            sub_spk_array = np.expand_dims(sub_spk_array, 1)
            sub_label = np.concatenate([sub_seg_array, sub_spk_array], axis=1)

            sub_label_filename = "{}/label/{}.rttm".format(args.tgt_dir, sub_filename)
            with open(sub_label_filename, 'w') as fh:
                for j in range(len(sub_label)):
                    fh.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(sub_filename, sub_label[j][0], sub_label[j][1] - sub_label[j][0], int(sub_label[j][2])))
        cnt += 1
        print("Finish {}/{} utts".format(cnt, len(uttlist)), flush=True)

        # debug mode
        if args.debug:
            if cnt > 3:
                break
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
