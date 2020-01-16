#!/usr/bin/python3
# This file filters out the utterances with one channel, and 
# only keeps utterances with two channels. The new uttname has 
# the format dataset-uttname-speaker_ch1-speaker_ch2-channel

import os
import sys

def utt_spk_mapping(filename):
    utt2spk_dict = {}
    spk2utt_dict = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        uttname = line.split()[0]
        spkname = line.split()[1]
        utt2spk_dict[uttname] = spkname
        if spkname in spk2utt_dict:
            spk2utt_dict[spkname].append(uttname)
        else:
            spk2utt_dict[spkname] = [uttname]
    return utt2spk_dict, spk2utt_dict

def utt_wav_mapping(filename):
    utt2wav_dict = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        uttname = line.split()[0]
        wav = " ".join(line.split()[1:])
        utt2wav_dict[uttname] = wav
    return utt2wav_dict

def filter_wav(filename, utt2wav_dict, out_dir):
    wav_scp_file = open("{}/wav.scp".format(out_dir), 'w')
    utt2spk_file = open("{}/utt2spk".format(out_dir), 'w')
    utt2singlechannel = {}
    with open(filename, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        uttname_single = line.split()[0]
        if line.split()[1] == "sox":
            continue
        speaker, dataset, uttname, channel = process_uttname(uttname_single)
        utt = "{}-{}".format(dataset, uttname)
        if utt in utt2singlechannel:
            utt2singlechannel[utt].append(uttname_single)
        else:
            utt2singlechannel[utt] = [uttname_single]
    total_utt = len(utt2singlechannel)
    utt_list = list(utt2singlechannel.keys())
    # remain the utterances if both the channels are in the dataset.
    for utt in utt_list:
        if len(utt2singlechannel[utt]) != 2:
            del utt2singlechannel[utt]
    remain_utt = len(utt2singlechannel)
    print("{} utts total, {} utts with two channels remain".format(total_utt, remain_utt))
    utt_list = list(utt2singlechannel.keys())
    utt_list.sort()
    for utt in utt_list:
        utt_singlechannel = utt2singlechannel[utt] 
        assert len(utt_singlechannel) == 2 # 2 channels
        if (utt_singlechannel[0])[-1] in ['a', 'A', '1']:
            utt_ch1 = utt_singlechannel[0]
            utt_ch2 = utt_singlechannel[1]
        elif (utt_singlechannel[0])[-1] in ['b', 'B', '2']:
            utt_ch1 = utt_singlechannel[1]
            utt_ch2 = utt_singlechannel[0]
        else:
            raise ValueError("Condition not defined.") 

        speaker_ch1, dataset_ch1, uttname_ch1, channel_ch1 = process_uttname(utt_ch1)
        speaker_ch2, dataset_ch2, uttname_ch2, channel_ch2 = process_uttname(utt_ch2)
        assert (dataset_ch1 == dataset_ch2) and (uttname_ch1 == uttname_ch2) and (channel_ch1 == "1") and (channel_ch2 == "2")
        utt_ch1_new = "{}-{}-{}-{}-{}".format(dataset_ch1, uttname_ch1, speaker_ch1, speaker_ch2, channel_ch1)
        utt_ch2_new = "{}-{}-{}-{}-{}".format(dataset_ch2, uttname_ch2, speaker_ch1, speaker_ch2, channel_ch2)
        wav_scp_file.write("{} {}\n".format(utt_ch1_new, utt2wav_dict[utt_ch1])) 
        wav_scp_file.write("{} {}\n".format(utt_ch2_new, utt2wav_dict[utt_ch2])) 
        utt2spk_file.write("{} {}\n".format(utt_ch1_new, utt_ch1_new))
        utt2spk_file.write("{} {}\n".format(utt_ch2_new, utt_ch2_new))
    wav_scp_file.close()
    utt2spk_file.close()
    return 0

def process_uttname(uttname):
    if len(uttname.split('_')) == 4 and uttname.split('_')[1] in ["MX6", "SRE08", "SRE10"]: 
        # Mixer6, SRE08, SRE10
        uttname_split = uttname.split('_')
        speaker = uttname_split[0]
        dataset = uttname_split[1]
        uttname = uttname_split[2]
        if uttname_split[3] == "A":
            channel = "1"
        elif uttname_split[3] == "B":
            channel = "2"
        else:
            print(uttname)
            raise ValueError("Channel not defined.")
    elif len(uttname.split('-')) == 4 and uttname.split('-')[1] in ["sre04", "sre05", "sre06"]: 
        # SRE04, 05, 06
        uttname_split = uttname.split('-')
        speaker = uttname_split[0]
        dataset = uttname_split[1]
        uttname = uttname_split[2]
        if uttname_split[3] == "a":
            channel = "1"
        elif uttname_split[3] == "b":
            channel = "2"
        else:
            print(uttname)
            raise ValueError("Channel not defined.")
    elif len(uttname.split('-')) == 2 and (uttname.split('-')[1]).split('_')[0] == "swbdc":
        # Switchboard Cellular Part 1, Switchboard Cellular Part 2
        speaker = uttname.split('-')[0]
        info = (uttname.split('-')[1]).split('_')
        assert len(info) == 4
        dataset = info[0]
        uttname = "_".join(info[1:3])
        channel = info[3]
        assert channel in ["1", "2"]
    elif len(uttname.split('_')) == 5 and uttname.split('_')[0] == "sw":
        # Switchboard-2 Phase I, Switchboard-2 Phase II, Switchboard-2 Phase III
        uttname_split = uttname.split('_')
        speaker = "_".join(uttname_split[:2])
        dataset = "swbd2"
        uttname = "_".join(uttname_split[2:4])
        channel = uttname_split[4]
        assert channel in ["1", "2"]
    else:
        print(uttname)
        raise ValueError("Condition not defined.")
    return speaker, dataset, uttname, channel

def main():
    swbd_sre_dir = sys.argv[1]
    out_dir = sys.argv[2]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    utt2wav_dict = utt_wav_mapping("{}/wav.scp".format(swbd_sre_dir))
    filter_wav("{}/wav.scp".format(swbd_sre_dir), utt2wav_dict, out_dir)
    return 0

if __name__ == "__main__":
    main()
