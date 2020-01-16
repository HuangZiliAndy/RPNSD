#!/usr/bin/env python3
# This script creates label.scp file for augmented data directory
# After data augmentation, the diarization label remains the same

import os
import sys

def load_label_scp(filename):
    utt2label = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt2label[line.split()[0]] = line.split()[1]
    return utt2label

def get_oriuttname(uttname):
    if uttname.endswith('music') or uttname.endswith('noise') or uttname.endswith('reverb') or uttname.endswith('babble'):
        return '-'.join(uttname.split('-')[:-1])
    else:
        return uttname

def main():
    data_dir = sys.argv[1]
    aug_dir = sys.argv[2]
    utt2label = load_label_scp("{}/label.scp".format(data_dir))
    label_scp_file = open("{}/label.scp".format(aug_dir), 'w')
    with open("{}/wav.scp".format(aug_dir), 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        uttname = line.split()[0]
        label_scp_file.write("{} {}\n".format(uttname, utt2label[get_oriuttname(uttname)]))
    label_scp_file.close()
    return 0

if __name__ == "__main__":
    main()
