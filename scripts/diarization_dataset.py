#!/usr/bin/env python3

# This script defines diarization dataset

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import feature
import kaldi_data
import random

random.seed(7)

# This function process the segment labels.
# It (1) merge two segments of the same speaker if their distance is
# smaller than merge_dis. (2) it removes the segments shorter than min_dis.
def process_seg(segment_array, merge_dis, min_dis):
    if len(segment_array) == 0:
        return segment_array
    spk_list = list(set(list(segment_array[:, 2])))
    seg_list = []
    for spk in spk_list:
        seg_spk = segment_array[segment_array[:, 2] == spk]
        seg_spk = seg_spk[seg_spk[:, 0].argsort()]
        new_seg = []
        for i in range(len(seg_spk)):
            if i == 0:
                new_seg.append(seg_spk[i, :])
            else:
                if seg_spk[i, 0] > new_seg[-1][1] + merge_dis:
                    new_seg.append(seg_spk[i, :])
                else:
                    new_seg[-1][1] = max(seg_spk[i, 1], new_seg[-1][1])
        new_seg = np.array(new_seg)
        seg_list.append(new_seg)
    seg_array = np.concatenate(seg_list, 0)
    seg_array = seg_array[seg_array[:, 0].argsort()]
    seg_array = seg_array[seg_array[:, 1] - seg_array[:, 0] > min_dis]
    return seg_array

class DiarDataset(Dataset):
    def __init__(self, data_dir, rate, frame_size, frame_shift, input_transform, padded_len, merge_dis, min_dis, num_utt=-1):
        # data_dir: Data directory
        # rate: Sample rate in Hz (default 8000)
        # frame_size: frame size (default 512)
        # frame_shift: frame shift (default 80)
        # input_transform: input transform to STFT feature. See scripts/feature.py for details
        # padded_len: max number of segments in a sample (default 20)
        # merge_dis: merge two segments if their distance is smaller than merge_dis (default 0)
        # min_dis: minimum length of each segment, discard segments that are too short (default 0.2)
        # num_utt: just use num_utt dev samples for validation, if num_utt < 0 use all of dev samples (default -1)

        self.data_dir = data_dir
        self.rate = rate
        self.uttlist = self.load_uttlist(data_dir)
        # sample part of the utterances
        if num_utt > 0:
            random.shuffle(self.uttlist)
            self.uttlist = self.uttlist[:num_utt]
            self.uttlist.sort()
        self.frame_size, self.frame_shift = frame_size, frame_shift
        self.input_transform = input_transform
        self.padded_len = padded_len
        self.merge_dis, self.min_dis = merge_dis, min_dis

    def __len__(self):
        return len(self.uttlist)

    def __getitem__(self, idx):
        uttname = self.uttlist[idx]
        info_filename = "{}/data/{}.txt".format(self.data_dir, uttname)
        assert os.path.exists(info_filename)
        with open(info_filename, 'r') as fh:
            info = fh.readline().strip('\n')
        info_split = info.split(None, 3)
        # each record has 4 fields
        # (1)uttname (2)uttdur (3)label file (4)feature file
        feat_file, label_file = info_split[3], info_split[2]

        # compute STFT feature
        data, samplerate = kaldi_data.load_wav(feat_file) 
        Y = feature.stft(data, self.frame_size, self.frame_shift)
        feat = feature.transform(Y, self.input_transform)

        # prepare diarization label
        label = self.process_label_file(label_file)
        second_per_frame = self.frame_shift * 1.0 / self.rate 
        label[:, :2] = (label[:, :2] / second_per_frame).astype(int)
        label[:, 2] = label[:, 2] + 1

        if len(label) > self.padded_len:
            print("Warning: length of {} exceeds padded length".format(uttname))
            label = label[:self.padded_len, :]
        label_padded = np.zeros((self.padded_len, 3))
        label_padded[:len(label), :] = label
        return uttname, feat, label_padded, len(label)

    def load_uttlist(self, data_dir):
        with open("{}/wav.scp".format(data_dir), 'r') as fh:
            content = fh.readlines()
        uttlist = []
        for line in content:
            line = line.strip('\n')
            uttlist.append(line.split()[0])
        uttlist.sort()
        return uttlist

    def process_label_file(self, label_filename):
        with open(label_filename, 'r') as fh:
            content = fh.readlines()
        label_list = []
        for line in content:
            line = line.strip('\n')
            line_split = line.split()
            start_t, duration, spkname = float(line_split[3]), float(line_split[4]), int(line_split[7])
            end_t = start_t + duration
            label_list.append([start_t, end_t, spkname])
        if len(label_list) == 0:
            segment_array = np.zeros((0, 3))
        else:
            segment_array = np.array(label_list)
        segment_array_new = process_seg(segment_array, self.merge_dis, self.min_dis)
        return segment_array_new

class DiarDataset_EVAL(Dataset):
    def __init__(self, data_dir, rate, frame_size, frame_shift, input_transform, merge_dis, min_dis):
        # data_dir: Data directory
        # rate: Sample rate in Hz (default 8000)
        # frame_size: frame size (default 512)
        # frame_shift: frame shift (default 80)
        # input_transform: input transform to STFT feature. See scripts/feature.py for details
        # merge_dis: merge two segments if their distance is smaller than merge_dis (default 0)
        # min_dis: minimum length of each segment, discard segments that are too short (default 0.2)
        self.data_dir = data_dir
        self.rate = rate
        self.utt2ark = self.load_wav_scp(data_dir)
        self.uttlist = list(self.utt2ark.keys())
        self.uttlist.sort()
        self.utt2seg = self.load_rttm(data_dir)
        self.frame_size, self.frame_shift = frame_size, frame_shift
        self.input_transform = input_transform
        self.merge_dis, self.min_dis = merge_dis, min_dis

    def __len__(self):
        return len(self.uttlist)

    def __getitem__(self, idx):
        uttname = self.uttlist[idx]
        data, samplerate = kaldi_data.load_wav(self.utt2ark[uttname]) 
        Y = feature.stft(data, self.frame_size, self.frame_shift)
        feat = feature.transform(Y, self.input_transform)

        seg_list = self.utt2seg[uttname]
        label = self.process_label(seg_list)
        second_per_frame = self.frame_shift * 1.0 / self.rate
        label[:, :2] = (label[:, :2] / second_per_frame).astype(int)
        label[:, 2] = label[:, 2] + 1
        return uttname, feat, label

    def load_wav_scp(self, data_dir):
        utt2ark = {}
        with open("{}/wav.scp".format(data_dir), 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            utt, ark = line.split(None, 1)
            utt2ark[utt] = ark
        return utt2ark

    def load_rttm(self, data_dir):
        utt2seg = {}
        with open("{}/rttm".format(data_dir), 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            uttname = line.split()[1]
            start_t, duration, spkname = float(line.split()[3]), float(line.split()[4]), line.split()[7]
            end_t = start_t + duration
            if uttname not in utt2seg:
                utt2seg[uttname] = []
            utt2seg[uttname].append([start_t, end_t, spkname])
        return utt2seg
    
    def process_label(self, seg_list):
        if len(seg_list) == 0:
            return np.zeros((0, 3))
        if len(seg_list) > 0:
            spk_list = list(set([seg[2] for seg in seg_list]))
            for i in range(len(seg_list)):
                seg_list[i][2] = spk_list.index(seg_list[i][2])
        seg_array = np.array(seg_list)
        seg_array_new = process_seg(seg_array, self.merge_dis, self.min_dis)
        return seg_array_new
