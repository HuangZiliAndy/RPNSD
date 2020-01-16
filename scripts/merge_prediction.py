# this script is called by eval_cpu.sh. It merges the 
# predictions on different machines

import os
import sys
import pickle
import argparse

parser = argparse.ArgumentParser(
    description='Merge prediction of different jobs')
parser.add_argument('predict_dir', type=str,
                    help='directory of prediction files')
parser.add_argument('nj', type=int,
                    help='number of jobs')

def main(args):
    predict_dict_all = {}
    for i in range(1, args.nj + 1):
        with open("{}/{}/detections.pkl".format(args.predict_dir, i), 'rb') as fh:
            predict_dict = pickle.load(fh)
            for key in predict_dict:
                predict_dict_all[key] = predict_dict[key]
    print("{} utts in total".format(len(predict_dict_all)))
    with open("{}/detections.pkl".format(args.predict_dir), 'wb') as fh:
        pickle.dump(predict_dict_all, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
