#!/usr/bin/env python3

# This script computes region proposals, confidence scores and speaker embeddings 
# with RPNSD model

import os
import torch
import argparse
import random
import sys
from diarization_dataset import DiarDataset_EVAL
import numpy as np
import socket
from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file
from utils import evaluate_no_nms
import pprint

print(socket.gethostname())

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(
    description='Region Proposal Network based Speaker Diarization Evaluation')
parser.add_argument('exp_dir', type=str,
                    help='path of experiment')
parser.add_argument('test_dir', type=str,
                    help='directory for test')
parser.add_argument('modelfile', type=str,
                    help='model filename')
parser.add_argument('--cfg_file', default="", type=str,
                    help='configure file')
parser.add_argument('--output_dir', default="", type=str,
                    help='output directory')

# data process parameters
parser.add_argument('--rate', default=8000, type=int,
                    help='sample rate')
parser.add_argument('--frame_size', default=512, type=int,
                    help='frame size')
parser.add_argument('--frame_shift', default=80, type=int,
                    help='frame shift')
parser.add_argument('--merge_dis', default=0.0, type=float,
                    help='merge two segments if their distance is smaller than merge_dis')
parser.add_argument('--min_dis', default=0.2, type=float,
                    help='minimum length of each segment, discard segments that are too short')
parser.add_argument('--padded_len', default=15, type=int,
                    help='label length after padding')

# training parameters
parser.add_argument('--batch_size', default=1, type=int,
                    help='mini-batch size')
parser.add_argument('--num_workers', default=0, type=int,
                    help='number of workers for data loading')
parser.add_argument('--seed', default=7, type=int,
                    help='random seed')
parser.add_argument('--freeze', default=0, type=int,
                    help='whether to freeze the model parameters')
parser.add_argument('--set_bn_fix', default=0, type=int,
                    help='whether to set batchnorm fixed')
parser.add_argument('--pretrain_resnet_model', default=None, type=str,
                    help='the directory of pretrained resnet model')
parser.add_argument('--use_gpu', default=0, type=int,
                    help='whether to use gpu for evaluation')

# network parameters
parser.add_argument('--arch', default='res101', type=str, 
                    help='model architecture')
parser.add_argument('--large_scale', default=0, type=int, 
                    help='whether use large imag scale')
parser.add_argument('--nclass', default=1284, type=int, 
                    help='number of classes (1283 spk and background)')

# validate parameters
parser.add_argument('--max_per_egs', default=20, type=int,
                    help='max number of segments in an example')
parser.add_argument('--cut_10s', default=0, type=int,
                    help='whether to cut the whole utterance into 10s chunks')
parser.add_argument('--save_per_chunk', default=0, type=int,
                    help='instead of returning predict results for whole utterance, return predict results for chunks')

def main():
    global args
    args = parser.parse_args()
    print(args)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_gpu:
        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # prepare test set
    test_dataset = DiarDataset_EVAL(args.test_dir, args.rate, args.frame_size, args.frame_shift, None, args.padded_len, args.merge_dis, args.min_dis)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False)
    print("{} TEST segments".format(len(test_dataset)))

    if args.cfg_file == "":
        args.cfg_file = "cfgs/{}_ls.yml".format(args.arch) if args.large_scale else "cfgs/{}.yml".format(args.arch)
    print("Using configure file {}".format(args.cfg_file))

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # initilize the network here.
    if args.arch == 'res101': 
        model = resnet(args.nclass, 101, pretrained=args.pretrain_resnet_model, freeze=args.freeze, set_bn_fix=args.set_bn_fix)
    else:
        print("network is not defined")
    model.create_architecture()
    model = model.to(device)

    # load parameters
    if os.path.isfile(args.modelfile):
        print("loading checkpoint '{}'".format(args.modelfile))
        if args.use_gpu:
            checkpoint = torch.load(args.modelfile)
        else:
            checkpoint = torch.load(args.modelfile, map_location="cpu")
        model.load_state_dict(checkpoint['model'])
        print("loaded checkpoint '{}' (epoch {} iter {})"
              .format(args.modelfile, checkpoint['epoch'], checkpoint['iter']))
        print("best score {:.4f}".format(checkpoint['best_score']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.modelfile))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # evaluation
    evaluate_no_nms(test_loader, model, device, args)
    return 0
            
if __name__ == "__main__":
    main()
