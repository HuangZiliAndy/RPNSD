#!/usr/bin/env python3

# This script performs post-processing for RPNSD predictions
# It mainly has two steps (1) clustering (2) NMS

import os
import numpy as np
import pickle
import argparse
import torch
from model.nms.nms_wrapper import nms
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans 

np.set_printoptions(suppress=True)

def get_args():
    parser = argparse.ArgumentParser(description="Post-processing for RPNSD outputs")
    parser.add_argument("predict_file", type=str,
                        help="Output file of faster RCNN model")
    parser.add_argument("rttm_file", type=str,
                        help="Output rttm file")
    parser.add_argument("--num_cluster", type=str, default=None, help="The reco2num_spk file")
    parser.add_argument("--cluster_type", type=str, default="kmeans", help="Threshold to make decisions")
    parser.add_argument("--thres", type=float, default=0.5, help="Threshold to make decisions")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="NMS threshold")
    parser.add_argument("--cluster_thres", type=float, default=0.7, help="Clustering threshold")
    parser.add_argument("--min_len", type=float, default=0,
                        help="Min length of segments")
    parser.add_argument("--merge_len", type=float, default=0,
                        help="Merge segments of the same class if their distance is smaller than merge_len")
    parser.add_argument("--remove_short", type=int, default=0,
                        help="Whether to remove short segments")
    parser.add_argument("--merge_seg", type=int, default=1,
                        help="Whether to merge segments")
    parser.add_argument("--rttm_channel", type=int, default=1,
                        help="RTTM channel")
  
    args = parser.parse_args()
    return args

# convert the detection output to diarization results
def post_process(utt2predict, args):
    utt2predict_new = {}
    for utt in utt2predict:
        # predict has the shape [*, 3]
        # (start_t, end_t, spk)
        predict = utt2predict[utt]
        # merge the segments of the same class
        if args.merge_seg:
            predict = merge_segments(predict, args.merge_len)
        # remove the segments that are too short
        if args.remove_short: 
            predict = predict[predict[:, 1] - predict[:, 0] > args.min_len]
        predict = predict[predict[:, 0].argsort()]
        utt2predict_new[utt] = predict
    return utt2predict_new

def merge_segments(seg_array, merge_dis):
    seg_array = seg_array[seg_array[:, 0].argsort()]
    spk_list = list(set(list(seg_array[:, 2])))
    seg_list = []
    for spk in spk_list:
        seg_spk = seg_array[seg_array[:, 2] == spk]
        seg_spk = merge_seg(seg_spk, merge_dis)
        seg_list.append(seg_spk)
    seg_array = np.concatenate(seg_list, 0)
    seg_array = seg_array[seg_array[:, 0].argsort()]
    return seg_array

def merge_seg(seg_array, merge_dis):
    seg_list = []
    for i in range(len(seg_array)):
        if i == 0:
            seg_list.append(seg_array[i, :])
        else:
            if seg_array[i, 0] > seg_list[-1][1] + merge_dis:
                seg_list.append(seg_array[i, :])
            else:
                seg_list[-1][1] = max(seg_list[-1][1], seg_array[i, 1])
    return np.array(seg_list)

def write_rttm(utt2seg, rttm_file, rttm_channel):
    uttlist = list(utt2seg.keys())
    uttlist.sort()
    with open(rttm_file, 'w') as fh:
        for utt in uttlist:
            seg = utt2seg[utt]
            for i in range(len(seg)):
                fh.write("SPEAKER {} {} {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(utt, rttm_channel, seg[i, 0], seg[i, 1] - seg[i, 0], int(seg[i, 2])))
    return 0

def apply_nms(utt2predict, nms_thres, device):
    # predict has the shape [*, 4]
    # (start_t, end_t, prob_bg, spk_label)
    utt2seg = {}
    uttlist = list(utt2predict.keys())
    for utt in uttlist:
        utt_predict = utt2predict[utt]
        spklist = list(set(utt_predict[:, 3]))
        spklist.sort()
        segments_list = []

        for spk in spklist:
            predict = utt_predict[utt_predict[:, 3] == spk, :]
            predict[:, :2] = (predict[:, :2] * 100.0).astype(int)
            predict = torch.from_numpy(predict).to(device)
            
            # apply nms
            # convert to 4 dim for NMS
            predict_input = torch.zeros(predict.size(0), 5).type_as(predict)
            predict_input[:, 0] = predict[:, 0]
            predict_input[:, 2] = predict[:, 1]
            predict_input[:, 4] = 1 - predict[:, 2]

            keep = nms(predict_input, nms_thres, force_cpu=True)
            segments = predict[keep.view(-1).long()].data.cpu().numpy()

            segments = segments[segments[:, 0].argsort()]
            segments = segments[:, [0, 1]]
            segments[:, :2] = segments[:, :2] / 100.0
            segments = np.insert(segments, 2, spk, axis=1)
            segments_list.append(segments)
        segments_array = np.concatenate(segments_list, axis=0)
        segments_array = segments_array[segments_array[:, 0].argsort()]
        utt2seg[utt] = segments_array
    return utt2seg

# perform clustering with speaker embeddings
def cluster(utt2predict, reco2num_spk, args):
    # prediction should have the shape [*, 3 + embedding_size]
    # (start_t, end_t, prob_bg, embedding)
    utt2predict_new = {}
    device = torch.device("cpu")

    uttlist = list(utt2predict.keys())
    uttlist.sort()
    for utt in uttlist:
        predict = utt2predict[utt]
        predict[:, :2] = predict[:, :2] / 100.0
        predict[:, 3:] = normalize(predict[:, 3:])
        # only keeps the segments with smaller background probability
        predict = predict[predict[:, 2] < args.thres]
        # remove the segments that are too short
        if args.remove_short: 
            predict = predict[predict[:, 1] - predict[:, 0] > args.min_len]
        predict = predict[predict[:, 0].argsort()]

        # clustering
        if args.cluster_type == "ahc":
            model = AgglomerativeClustering(n_clusters=reco2num_spk[utt], affinity="euclidean", compute_full_tree=True, linkage="ward")
        elif args.cluster_type == "spec":
            model = SpectralClustering(n_clusters=reco2num_spk[utt], assign_labels="discretize", random_state=0)
        elif args.cluster_type == "kmeans":
            model = KMeans(n_clusters=reco2num_spk[utt], random_state=0)
        else:
            raise ValueError("Condition not defined.")
        labels = model.fit_predict(predict[:, 3:])
        labels = np.expand_dims(labels, axis=1)
        predict_new = np.concatenate([predict[:, :3], labels], axis=1)
        utt2predict_new[utt] = predict_new

    return utt2predict_new

def load_reco2num_spk(filename):
    reco2num_spk = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname, num_spk = line_split[0], int(line_split[1])
        reco2num_spk[uttname] = num_spk
    return reco2num_spk

def main():
    args = get_args()

    # load predict file
    # prediction should have the shape [*, 3 + embedding_size]
    # (start_t, end_t, prob_bg, embedding)
    with open(args.predict_file, 'rb') as fh:
        utt2predict = pickle.load(fh)

    # load reco2num_spk file
    if args.num_cluster:
        reco2num_spk = load_reco2num_spk(args.num_cluster)
    else:
        # Our current setup only supports clustering with known
        # number of speakers. Of course, this information may not
        # exist for many situations. However, you can use some 
        # threshold based methods (like AHC) to deal with these situations
        raise ValueError("{} is required".format(args.num_cluster))

    # perform clustering
    utt2predict_cluster = cluster(utt2predict, reco2num_spk, args)

    # apply nms for same class
    utt2seg = apply_nms(utt2predict_cluster, args.nms_thres, torch.device("cpu"))

    # perform some post processing
    # (1) merge two segments of the same speaker if the distance between
    # them are shorter than merge_len
    # (2) remove the segments that are too short
    utt2seg = post_process(utt2seg, args)

    # write RTTM file
    write_rttm(utt2seg, args.rttm_file, args.rttm_channel)
    return 0

if __name__ == "__main__":
    main()
