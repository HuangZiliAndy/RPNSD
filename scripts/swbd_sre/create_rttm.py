#!/usr/bin/env python3

import os
import sys
import argparse

# This script creates the RTTM file based on the segments file.
# The RTTM file has the format
# SPEAKER iaaa 0 6.14 2.99 <NA> <NA> B <NA> <NA>

parser = argparse.ArgumentParser("Create RTTM file from segments file for SWBD SRE dataset")
parser.add_argument('segments_file', type=str, help='segments file')
parser.add_argument('output_dir', type=str, help='output directory')

def create_rttm(segments_file, output_dir):
    utt_list = []
    rttm_file = open("{}/rttm".format(output_dir), 'w')
    with open(segments_file, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        uttname = line_split[1]
        starttime = float(line_split[2])
        endtime = float(line_split[3])
        duration = endtime - starttime
        uttname_split = uttname.split('-')
        assert len(uttname_split) == 5
        new_uttname = "-".join(uttname_split[:-1])

        if len(utt_list)==0 or utt_list[-1] != new_uttname:
            utt_list.append(new_uttname)

        if uttname_split[-1] == '1': # channel 1
            spkname = uttname_split[2]
        elif uttname_split[-1] == '2': # channel 2
            spkname = uttname_split[3]
        else:
            raise ValueError("Condition not defined.")
        rttm_file.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(new_uttname, starttime, duration, spkname))
    rttm_file.close()
    return utt_list

def main(args):
    # create RTTM file
    utt_list = create_rttm(args.segments_file, args.output_dir)
    print("Finish preparing RTTM file for {} utterances".format(len(utt_list)))
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
