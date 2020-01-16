#!/usr/bin/env python3

# This script is called by split_utt.sh, and it creates the wav.scp file

import os
import sys

def main():
    data_dir = sys.argv[1]
    abs_data_dir = os.path.abspath(data_dir)

    assert os.path.exists("{}/wav".format(data_dir))
    wav_file_list = os.listdir("{}/wav".format(data_dir))
    wav_file_list.sort()

    with open("{}/wav.scp".format(data_dir), 'w') as fh:
        for wav_file in wav_file_list:
            uttname = wav_file.split('.')[0]
            fh.write("{} {}/wav/{}\n".format(uttname, abs_data_dir, wav_file))
    return 0

if __name__ == "__main__":
    main()
