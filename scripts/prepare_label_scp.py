#!/usr/bin/env python3

# This script is called by split_utt.sh, and it creates the label.scp file

import os
import sys

def main():
    data_dir = sys.argv[1]
    abs_data_dir = os.path.abspath(data_dir)

    assert os.path.exists("{}/label".format(data_dir))
    label_file_list = os.listdir("{}/label".format(data_dir))
    label_file_list.sort()

    with open("{}/label.scp".format(data_dir), 'w') as fh:
        for label_file in label_file_list:
            uttname = label_file.split('.')[0]
            fh.write("{} {}/label/{}\n".format(uttname, abs_data_dir, label_file))
    return 0

if __name__ == "__main__":
    main()
